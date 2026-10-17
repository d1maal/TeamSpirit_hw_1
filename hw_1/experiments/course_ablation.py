from pathlib import Path
import hashlib
import itertools
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from . import course_ovr as g
RESULT_DIR = Path('artifacts/course_ablation')
VARIANTS = ['full', 'without_next', 'without_previous', 'without_structure', 'shared_text']
ALPHAS = [0.0, 0.5, 1.0]


def features(dialogs, training=False):
    pairs = g.interactions(dialogs, training)
    pairs['shared'] = pairs.user + '\n' + pairs.assistant + '\n' + pairs.following
    return pairs


def encoder(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if variant == 'full':
        return g.encoder(True)
    base = g.encoder(variant != 'without_next')
    if variant == 'without_previous':
        base.transformers = [b for b in base.transformers if b[0] != 'assistant']
        base.transformer_weights.pop('assistant')
    elif variant == 'without_structure':
        base.transformers = [b for b in base.transformers if b[0] != 'structure']
        base.transformer_weights.pop('structure')
    elif variant == 'shared_text':
        word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=24000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        char = next((b[1] for b in base.transformers if b[0] == 'user_char'))
        base = ColumnTransformer(
            [('shared_word', word, 'shared'), ('user_char', char, 'user'), ('structure', StandardScaler(with_mean=False), ['position', 'log_length', 'n_users', 'questions', 'exclamations'])],
            transformer_weights={'shared_word': 1.0, 'user_char': 1.0, 'structure': 0.1},
            sparse_threshold=1.0,
        )
    return base


def aggregate(pairs, positive, negative, n_dialogs, thresholds, alphas=(0.0, 0.0)):
    position = pairs.position.to_numpy()
    return g.aggregate(pairs, positive - alphas[0] * position, negative - alphas[1] * position, n_dialogs, thresholds)


def inner_scores(training, variant, configs, inner_splits):
    pairs = features(training, True)
    scores = {label: np.full((len(configs), len(pairs)), np.nan) for label in ['positive', 'negative']}
    cv = StratifiedGroupKFold(inner_splits, shuffle=True, random_state=42)
    for tr, va in cv.split(training, training.reaction, training.group):
        assert not set(training.group.iloc[tr]) & set(training.group.iloc[va])
        train = features(training.iloc[tr], True)
        valid = features(training.iloc[va].drop(columns=['labels', 'reaction']))
        enc = encoder(variant)
        xt = enc.fit_transform(train)
        xv = enc.transform(valid)
        mask = pairs.dialog_pos.isin(va).to_numpy()
        assert mask.sum() == len(valid)
        for label in scores:
            for k, config in enumerate(configs):
                model, scale = g.fit_binary(xt, train.reaction.to_numpy(), label, config)
                scores[label][k, mask] = model.decision_function(xv) / scale
    assert all((np.isfinite(v).all() for v in scores.values()))
    return (pairs, scores)


def macro(y, pred):
    return float(np.mean([2 * np.sum((y == c) & (pred == c)) / max(np.sum(y == c) + np.sum(pred == c), 1) for c in g.CLASSES]))


def choose(training, pairs, scores, configs, position_threshold):
    best = None
    rows = []
    y = training.reaction.to_numpy()
    slopes = ALPHAS if position_threshold else [0.0]
    for kp, kn, tp, tn, ap, an in itertools.product(range(len(configs)), range(len(configs)), g.THRESHOLDS, g.THRESHOLDS, slopes, slopes):
        pred = aggregate(
            pairs,
            scores['positive'][kp],
            scores['negative'][kn],
            len(training),
            (tp, tn),
            (ap, an),
        )
        value = macro(y, pred)
        rows.append((kp, kn, tp, tn, ap, an, value))
        if best is None or value > best['inner_macro_f1'] + 1e-12:
            best = dict(
                positive=configs[kp],
                negative=configs[kn],
                thresholds=(tp, tn),
                alphas=(ap, an),
                inner_macro_f1=value,
            )
    return (best, pd.DataFrame(rows, columns=['positive_config', 'negative_config', 'tp', 'tn', 'alpha_p', 'alpha_n', 'macro_f1']))


def fit(training, variant, choice):
    pairs = features(training, True)
    enc = encoder(variant)
    x = enc.fit_transform(pairs)
    models = {label: g.fit_binary(x, pairs.reaction.to_numpy(), label, choice[label]) for label in ['positive', 'negative']}
    return dict(encoder=enc, models=models, variant=variant, choice=choice, train_dialog_ids=training.dialog_id.tolist())


def predict(bundle, dialogs):
    pairs = features(dialogs[['dialog_id', 'turns']].reset_index(drop=True))
    if pairs.empty:
        return np.full(len(dialogs), 'neutral', dtype=object)
    x = bundle['encoder'].transform(pairs)
    s = {label: model.decision_function(x) / scale for label, (model, scale) in bundle['models'].items()}
    choice = bundle['choice']
    return aggregate(pairs, s['positive'], s['negative'], len(dialogs), choice['thresholds'], choice['alphas'])


def fingerprint(training, configs, outer_splits, inner_splits):
    payload = training[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records')
    return hashlib.sha256(Path(__file__).read_bytes() + Path(g.__file__).read_bytes() + payload.encode() + repr((configs, outer_splits, inner_splits)).encode()).hexdigest()


def run(training, configs=g.CONFIGS, outer_splits=5, inner_splits=3):
    training = training.reset_index(drop=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    key = fingerprint(training, configs, outer_splits, inner_splits)
    destination = RESULT_DIR / key[:16]
    destination.mkdir(exist_ok=True)
    names = VARIANTS + ['full_position_threshold']
    predictions = {name: np.full(len(training), '', dtype=object) for name in names}
    folds = []
    choices = []
    cv = StratifiedGroupKFold(outer_splits, shuffle=True, random_state=42)
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(cv.split(training, training.reaction, training.group), 1):
            assert not set(training.group.iloc[tr]) & set(training.group.iloc[va])
            train = training.iloc[tr].reset_index(drop=True)
            for variant in VARIANTS:
                cache = destination / f'{variant}_fold{fold}.joblib'
                if cache.exists():
                    data = joblib.load(cache)
                else:
                    pairs, scores = inner_scores(train, variant, configs, inner_splits)
                    data = {}
                    for positional in [False, True] if variant == 'full' else [False]:
                        name = variant if not positional else 'full_position_threshold'
                        choice, search = choose(train, pairs, scores, configs, positional)
                        search.to_csv(destination / f'{name}_fold{fold}_search.csv', index=False)
                        bundle = fit(train, variant, choice)
                        pred = predict(bundle, training.iloc[va])
                        data[name] = dict(pred=pred, choice=choice)
                    joblib.dump(data, cache)
                for name, values in data.items():
                    predictions[name][va] = values['pred']
                    score = macro(training.reaction.to_numpy()[va], values['pred'])
                    folds.append(dict(variant=name, fold=fold, macro_f1=score))
                    choices.append(dict(variant=name, fold=fold, **values['choice']))
    assert all(((v != '').all() for v in predictions.values()))
    fold_table = pd.DataFrame(folds)
    table = pd.DataFrame([dict(variant=name, OOF_Macro_F1=macro(training.reaction.to_numpy(), pred), Mean_CV_Macro_F1=fold_table[fold_table.variant == name].macro_f1.mean()) for name, pred in predictions.items()]).set_index('variant')
    table['Delta_vs_full'] = table.OOF_Macro_F1 - table.loc['full', 'OOF_Macro_F1']
    paired = fold_table.pivot(index='fold', columns='variant', values='macro_f1')
    paired = paired.subtract(paired['full'], axis=0)
    result = dict(
        fingerprint=key,
        table=table,
        fold_scores=fold_table,
        paired_deltas=paired,
        predictions=predictions,
        choices=choices,
        y=training.reaction.to_numpy(),
        dialog_ids=training.dialog_id.tolist(),
        directory=str(destination),
    )
    table.to_csv(destination / 'comparison.csv')
    paired.to_csv(destination / 'paired_fold_deltas.csv')
    joblib.dump(result, destination / 'results.joblib')
    return result


def train_final(training, name, configs=g.CONFIGS, inner_splits=3):
    variant = 'full' if name == 'full_position_threshold' else name
    with threadpool_limits(limits=1):
        pairs, scores = inner_scores(training, variant, configs, inner_splits)
        choice, _ = choose(training, pairs, scores, configs, name == 'full_position_threshold')
        return fit(training, variant, choice)
