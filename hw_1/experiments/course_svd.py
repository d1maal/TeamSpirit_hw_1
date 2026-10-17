from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from . import course_ovr as g
from . import course_ablation as a
RESULT_DIR = Path('artifacts/course_svd')
DIMENSIONS = [100, 200, 300]


def encoder(dimension):
    tfidf = FeatureUnion([('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True, dtype=np.float32)), ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=18000, sublinear_tf=True, dtype=np.float32))])
    steps = [('tfidf', tfidf)]
    if dimension is not None:
        steps.append(('original_and_svd', FeatureUnion([('original', FunctionTransformer(accept_sparse=True)), ('svd', TruncatedSVD(n_components=dimension, n_iter=5, random_state=42))])))
    return ColumnTransformer([('user', Pipeline(steps), 'user')], sparse_threshold=1.0)


def inner_scores(training, dimension, configs, inner_splits):
    pairs = g.interactions(training, True)
    scores = {label: np.full((len(configs), len(pairs)), np.nan) for label in ['positive', 'negative']}
    cv = StratifiedGroupKFold(inner_splits, shuffle=True, random_state=42)
    for tr, va in cv.split(training, training.reaction, training.group):
        assert not set(training.group.iloc[tr]) & set(training.group.iloc[va])
        pt = g.interactions(training.iloc[tr], True)
        pv = g.interactions(training.iloc[va].drop(columns=['labels', 'reaction']))
        enc = encoder(dimension)
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


def fit(training, dimension, choice):
    pairs = g.interactions(training, True)
    enc = encoder(dimension)
    x = enc.fit_transform(pairs)
    models = {label: g.fit_binary(x, pairs.reaction.to_numpy(), label, choice[label]) for label in ['positive', 'negative']}
    return dict(encoder=enc, models=models, dimension=dimension, choice=choice, train_dialog_ids=training.dialog_id.tolist())


def predict(bundle, dialogs):
    pairs = g.interactions(dialogs[['dialog_id', 'turns']].reset_index(drop=True))
    if pairs.empty:
        return np.full(len(dialogs), 'neutral', dtype=object)
    x = bundle['encoder'].transform(pairs)
    s = {label: model.decision_function(x) / scale for label, (model, scale) in bundle['models'].items()}
    return g.aggregate(pairs, s['positive'], s['negative'], len(dialogs), bundle['choice']['thresholds'])


def fingerprint(training, dimensions, configs, outer_splits, inner_splits):
    payload = training[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records')
    source = Path(__file__).read_bytes() + Path(g.__file__).read_bytes() + Path(a.__file__).read_bytes()
    return hashlib.sha256(source + payload.encode() + repr((dimensions, configs, outer_splits, inner_splits)).encode()).hexdigest()


def select(training, dimensions, configs, inner_splits, destination):
    selected = {}
    for dimension in [None] + list(dimensions):
        pairs, scores = inner_scores(training, dimension, configs, inner_splits)
        choice, search = a.choose(training, pairs, scores, configs, False)
        search.to_csv(destination / f'search_dim_{dimension}.csv', index=False)
        name = 'user_tfidf' if dimension is None else 'user_tfidf_plus_svd'
        if name not in selected or choice['inner_macro_f1'] > selected[name][1]['inner_macro_f1']:
            selected[name] = (dimension, choice)
    return selected


def run(training, dimensions=DIMENSIONS, configs=g.CONFIGS, outer_splits=5, inner_splits=3):
    training = training.reset_index(drop=True)
    key = fingerprint(training, dimensions, configs, outer_splits, inner_splits)
    destination = RESULT_DIR / key[:16]
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'results.joblib').exists():
        return joblib.load(destination / 'results.joblib')
    predictions = {name: np.full(len(training), '', dtype=object) for name in ['user_tfidf', 'user_tfidf_plus_svd']}
    folds = []
    choices = []
    cv = StratifiedGroupKFold(outer_splits, shuffle=True, random_state=42)
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(cv.split(training, training.reaction, training.group), 1):
            assert not set(training.group.iloc[tr]) & set(training.group.iloc[va])
            fold_dir = destination / f'fold_{fold}'
            fold_dir.mkdir(exist_ok=True)
            cache = fold_dir / 'predictions.joblib'
            if cache.exists():
                data = joblib.load(cache)
            else:
                train = training.iloc[tr].reset_index(drop=True)
                selected = select(train, dimensions, configs, inner_splits, fold_dir)
                data = {}
                for name, (dimension, choice) in selected.items():
                    bundle = fit(train, dimension, choice)
                    data[name] = dict(pred=predict(bundle, training.iloc[va]), dimension=dimension, choice=choice)
                joblib.dump(data, cache)
            for name, v in data.items():
                predictions[name][va] = v['pred']
                score = a.macro(training.reaction.to_numpy()[va], v['pred'])
                folds.append(dict(variant=name, fold=fold, macro_f1=score))
                choices.append(dict(variant=name, fold=fold, dimension=v['dimension'], **v['choice']))
    fold_table = pd.DataFrame(folds)
    table = pd.DataFrame([dict(variant=name, OOF_Macro_F1=a.macro(training.reaction.to_numpy(), pred), Mean_CV_Macro_F1=fold_table[fold_table.variant == name].macro_f1.mean()) for name, pred in predictions.items()]).set_index('variant')
    table['Delta_vs_user_tfidf'] = table.OOF_Macro_F1 - table.loc['user_tfidf', 'OOF_Macro_F1']
    result = dict(
        table=table,
        predictions=predictions,
        y=training.reaction.to_numpy(),
        choices=choices,
        fold_scores=fold_table,
        dialog_ids=training.dialog_id.tolist(),
        fingerprint=key,
    )
    table.to_csv(destination / 'comparison.csv')
    joblib.dump(result, destination / 'results.joblib')
    return result


def train_final(training, dimensions=DIMENSIONS, configs=g.CONFIGS, inner_splits=3):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        selected = select(training.reset_index(drop=True), dimensions, configs, inner_splits, RESULT_DIR)
        dimension, choice = selected['user_tfidf_plus_svd']
        return fit(training, dimension, choice)
