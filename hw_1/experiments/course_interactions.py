from pathlib import Path
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
import itertools
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from scipy import sparse
from threadpoolctl import threadpool_limits
from . import course_ovr as g
from . import course_ablation as a
RESULT_DIR = Path('artifacts/course_interactions')
VARIANTS = ['G_control', 'G_signals', 'G_interactions']
PATTERNS = {
    'negation': r"\b(?:not|no|never|cannot|can't|don't|doesn't|didn't|isn't|wasn't)\b",
    'disagreement': r"\b(?:wrong|incorrect|false|inaccurate|disagree|mistake|error)\b",
    'revision': r"\b(?:fix|correct|revise|rewrite|redo|change|instead|still)\b",
    'clarification': r"\b(?:explain|clarify|elaborate|detail|mean|understand)\b",
    'gratitude': r"\b(?:thanks|thank you|grateful|appreciate)\b",
    'approval': r"\b(?:great|perfect|excellent|helpful|works|awesome)\b",
    'apology': r"\b(?:sorry|apologize|apologise|apologies)\b",
    'admission': r"\b(?:you(?:'re| are) right|my mistake|i was wrong|i made (?:a |an )?(?:mistake|error)|i misunderstood)\b",
    'welcome': r"\b(?:you(?:'re| are) welcome|glad (?:to|it)|happy to help)\b",
}
TOKEN = re.compile(r"\b\w+\b")


class InteractionSignals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array([f'{source}_{name}' for source in ['user', 'assistant', 'following'] for name in PATTERNS] + ['overlap_user_previous', 'overlap_user_following', 'overlap_assistants'], dtype=object)

    def transform(self, x):
        texts = {source: x[source].fillna('').str.lower().tolist() for source in ['user', 'assistant', 'following']}
        columns = []
        for source in texts:
            for pattern in PATTERNS.values():
                regex = re.compile(pattern)
                columns.append([float(bool(regex.search(t))) for t in texts[source]])
        tokens = {source: [set(TOKEN.findall(t)) for t in seq] for source, seq in texts.items()}
        for left, right in [('user', 'assistant'), ('user', 'following'), ('assistant', 'following')]:
            columns.append([len(u & v) / max(len(u | v), 1) for u, v in zip(tokens[left], tokens[right])])
        return sparse.csr_matrix(np.asarray(columns, dtype=np.float32).T)


def encoder(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    base = g.encoder(True)
    if variant == 'G_control':
        return base
    steps = [('signals', InteractionSignals())]
    if variant == 'G_interactions':
        steps.append(('pairwise', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
    steps.append(('scale', StandardScaler(with_mean=False)))
    return FeatureUnion([('G', base), ('signals', Pipeline(steps))], transformer_weights={'G': 1.0, 'signals': 0.1})


def inner_scores(training, variant, configs, inner_splits):
    pairs = g.interactions(training, True)
    scores = {label: np.full((len(configs), len(pairs)), np.nan) for label in ['positive', 'negative']}
    cv = StratifiedGroupKFold(inner_splits, shuffle=True, random_state=42)
    for tr, va in cv.split(training, training.reaction, training.group):
        assert not set(training.group.iloc[tr]) & set(training.group.iloc[va])
        pt = g.interactions(training.iloc[tr], True)
        pv = g.interactions(training.iloc[va].drop(columns=['labels', 'reaction']))
        enc = encoder(variant)
        xt = enc.fit_transform(pt)
        xv = enc.transform(pv)
        mask = pairs.dialog_pos.isin(va).to_numpy()
        assert mask.sum() == len(pv)
        for label in scores:
            for k, config in enumerate(configs):
                model, scale = g.fit_binary(xt, pt.reaction.to_numpy(), label, config)
                scores[label][k, mask] = model.decision_function(xv) / scale
    assert all((np.isfinite(s).all() for s in scores.values()))
    return (pairs, scores)


def fit(training, variant, choice):
    pairs = g.interactions(training, True)
    enc = encoder(variant)
    x = enc.fit_transform(pairs)
    models = {label: g.fit_binary(x, pairs.reaction.to_numpy(), label, choice[label]) for label in ['positive', 'negative']}
    return dict(encoder=enc, models=models, variant=variant, choice=choice, train_dialog_ids=training.dialog_id.tolist())


def predict(bundle, dialogs):
    pairs = g.interactions(dialogs[['dialog_id', 'turns']].reset_index(drop=True))
    if pairs.empty:
        return np.full(len(dialogs), 'neutral', dtype=object)
    x = bundle['encoder'].transform(pairs)
    scores = {label: model.decision_function(x) / scale for label, (model, scale) in bundle['models'].items()}
    return g.aggregate(pairs, scores['positive'], scores['negative'], len(dialogs), bundle['choice']['thresholds'])


def fingerprint(training, configs, outer_splits, inner_splits):
    source = Path(__file__).read_bytes() + Path(g.__file__).read_bytes() + Path(a.__file__).read_bytes()
    payload = training[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records')
    return hashlib.sha256(source + payload.encode() + repr((configs, outer_splits, inner_splits)).encode()).hexdigest()


def matrix_variants(train_pairs, valid_pairs, names):
    base = g.encoder(True)
    xt = base.fit_transform(train_pairs)
    xv = base.transform(valid_pairs)
    for name in names:
        if name == 'G_control':
            yield (name, xt, xv)
        else:
            extra = dict(encoder(name).transformer_list)['signals']
            et = extra.fit_transform(train_pairs) * 0.1
            ev = extra.transform(valid_pairs) * 0.1
            yield (name, sparse.hstack([xt, et], format='csr'), sparse.hstack([xv, ev], format='csr'))


def fit_scores(task, xt, xv, reactions):
    label, k, config = task
    model, scale = g.fit_binary(xt, reactions, label, config)
    return (label, k, model.decision_function(xv) / scale)


def parallel_scores(xt, xv, reactions, configs, n_jobs):
    tasks = [(label, k, config) for label in ['positive', 'negative'] for k, config in enumerate(configs)]
    if n_jobs == 1:
        return [fit_scores(t, xt, xv, reactions) for t in tasks]
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(lambda task: fit_scores(task, xt, xv, reactions), tasks))


def choose_fast(training, pairs, scores, configs, batch_size=32):
    grid = list(itertools.product(range(len(configs)), range(len(configs)), g.THRESHOLDS, g.THRESHOLDS))
    ids = pairs.dialog_pos.to_numpy(dtype=int)
    counts = np.bincount(ids, minlength=len(training))
    present = np.flatnonzero(counts)
    starts = np.cumsum(np.r_[0, counts])[:-1][present]
    y = np.array([g.CLASSES.index(v) for v in training.reaction])
    values = []
    indices = np.arange(len(pairs))
    for offset in range(0, len(grid), batch_size):
        batch = grid[offset:offset + batch_size]
        kp, kn, tp, tn = map(np.asarray, zip(*batch))
        ep = scores['positive'][kp] - tp[:, None]
        en = scores['negative'][kn] - tn[:, None]
        found = (ep >= 0) | (en >= 0)
        last = np.maximum.reduceat(np.where(found, indices[None, :], -1), starts, axis=1)
        pred = np.ones((len(batch), len(training)), dtype=np.int8)
        rows, cols = np.nonzero(last >= 0)
        loc = last[rows, cols]
        pred[rows, present[cols]] = np.where(ep[rows, loc] > en[rows, loc], 2, 0)
        f1 = np.zeros(len(batch))
        for code in range(3):
            truth = y == code
            guessed = pred == code
            tp_count = np.sum(guessed & truth[None, :], axis=1)
            denominator = np.sum(guessed, axis=1) + truth.sum()
            f1 += np.divide(2 * tp_count, denominator, out=np.zeros(len(batch)), where=denominator > 0) / 3
        values.extend(f1.tolist())
    best = 0
    for k, value in enumerate(values):
        if value > values[best] + 1e-12:
            best = k
    kp, kn, tp, tn = grid[best]
    choice = dict(
        positive=configs[kp],
        negative=configs[kn],
        thresholds=(tp, tn),
        alphas=(0.0, 0.0),
        inner_macro_f1=values[best],
    )
    rows = [(*entry, 0.0, 0.0, value) for entry, value in zip(grid, values)]
    return (choice, pd.DataFrame(rows, columns=['positive_config', 'negative_config', 'tp', 'tn', 'alpha_p', 'alpha_n', 'macro_f1']))


def inner_scores_shared(training, names, configs, inner_splits, n_jobs=2):
    pairs = g.interactions(training, True)
    all_scores = {name: {label: np.full((len(configs), len(pairs)), np.nan) for label in ['positive', 'negative']} for name in names}
    cv = StratifiedGroupKFold(inner_splits, shuffle=True, random_state=42)
    for tr, va in cv.split(training, training.reaction, training.group):
        assert not set(training.group.iloc[tr]) & set(training.group.iloc[va])
        pt = g.interactions(training.iloc[tr], True)
        pv = g.interactions(training.iloc[va].drop(columns=['labels', 'reaction']))
        mask = pairs.dialog_pos.isin(va).to_numpy()
        assert mask.sum() == len(pv)
        for name, xt, xv in matrix_variants(pt, pv, names):
            for label, k, values in parallel_scores(xt, xv, pt.reaction.to_numpy(), configs, n_jobs):
                all_scores[name][label][k, mask] = values
    assert all((np.isfinite(v).all() for scores in all_scores.values() for v in scores.values()))
    return (pairs, all_scores)


def run(training, configs=g.CONFIGS, outer_splits=5, inner_splits=3, n_jobs=2):
    if n_jobs not in (1, 2, 3, 4):
        raise ValueError('n_jobs must be 1, 2, 3 or 4')
    training = training.reset_index(drop=True)
    key = fingerprint(training, configs, outer_splits, inner_splits)
    destination = RESULT_DIR / key[:16]
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'results.joblib').exists():
        return joblib.load(destination / 'results.joblib')
    predictions = {name: np.full(len(training), '', dtype=object) for name in VARIANTS}
    folds = []
    choices = []
    cv = StratifiedGroupKFold(outer_splits, shuffle=True, random_state=42)
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(cv.split(training, training.reaction, training.group), 1):
            assert not set(training.group.iloc[tr]) & set(training.group.iloc[va])
            train = training.iloc[tr].reset_index(drop=True)
            missing = [name for name in VARIANTS if not (destination / f'{name}_fold{fold}.joblib').exists()]
            if missing:
                pairs, all_scores = inner_scores_shared(train, missing, configs, inner_splits, n_jobs)
                selected = {}
                for name in missing:
                    choice, search = choose_fast(train, pairs, all_scores[name], configs)
                    selected[name] = choice
                    search.to_csv(destination / f'{name}_fold{fold}_search.csv', index=False)
                del all_scores
                pt = g.interactions(train, True)
                pv = g.interactions(training.iloc[va].drop(columns=['labels', 'reaction']))
                for name, xt, xv in matrix_variants(pt, pv, missing):
                    choice = selected[name]
                    scores = {}
                    for label in ['positive', 'negative']:
                        model, scale = g.fit_binary(xt, pt.reaction.to_numpy(), label, choice[label])
                        scores[label] = model.decision_function(xv) / scale
                    pred = g.aggregate(
                        pv,
                        scores['positive'],
                        scores['negative'],
                        len(va),
                        choice['thresholds'],
                    )
                    joblib.dump(dict(pred=pred, choice=choice), destination / f'{name}_fold{fold}.joblib')
            for name in VARIANTS:
                data = joblib.load(destination / f'{name}_fold{fold}.joblib')
                predictions[name][va] = data['pred']
                value = a.macro(training.reaction.to_numpy()[va], data['pred'])
                folds.append(dict(variant=name, fold=fold, macro_f1=value))
                choices.append(dict(variant=name, fold=fold, **data['choice']))
    assert all(((p != '').all() for p in predictions.values()))
    fold_table = pd.DataFrame(folds)
    table = pd.DataFrame([dict(variant=name, OOF_Macro_F1=a.macro(training.reaction.to_numpy(), pred), Mean_CV_Macro_F1=fold_table[fold_table.variant == name].macro_f1.mean()) for name, pred in predictions.items()]).set_index('variant')
    table['Delta_vs_G'] = table.OOF_Macro_F1 - table.loc['G_control', 'OOF_Macro_F1']
    paired = fold_table.pivot(index='fold', columns='variant', values='macro_f1')
    paired['interactions_minus_signals'] = paired.G_interactions - paired.G_signals
    paired['signals_minus_G'] = paired.G_signals - paired.G_control
    result = dict(
        table=table,
        paired=paired,
        fold_scores=fold_table,
        choices=choices,
        predictions=predictions,
        y=training.reaction.to_numpy(),
        dialog_ids=training.dialog_id.tolist(),
        fingerprint=key,
    )
    table.to_csv(destination / 'comparison.csv')
    paired.to_csv(destination / 'paired_folds.csv')
    joblib.dump(result, destination / 'results.joblib')
    return result


def train_final(training, variant='G_interactions', configs=g.CONFIGS, inner_splits=3):
    with threadpool_limits(limits=1):
        pairs, scores = inner_scores(training, variant, configs, inner_splits)
        choice, _ = choose_fast(training, pairs, scores, configs)
        return fit(training, variant, choice)
