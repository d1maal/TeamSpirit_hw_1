from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import re
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from . import course_ovr as g, course_interactions as j, course_ablation as a
RESULT_DIR = Path('artifacts/course_decision_weight')
POLICIES = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 2.0, 2.0)]
POLICY_NAMES = {(1.0, 1.0, 1.0): 'G_signals', (2.0, 2.0, 2.0): 'Decision_2x', (3.0, 2.0, 2.0): 'Decision_3x'}


def sample_weights(pairs, policy, normalize_dialogues=False):
    y = pairs.reaction.to_numpy()
    ids = pairs.dialog_pos.to_numpy(dtype=int)
    weights = np.ones(len(pairs), dtype=float)
    for dialog in np.unique(ids):
        indices = np.flatnonzero(ids == dialog)
        nn = indices[y[indices] != 'neutral']
        if len(nn):
            decisive = nn[-1]
            weights[decisive] = policy[0]
            late = indices[(indices > decisive) & (y[indices] == 'neutral')]
            weights[late] = policy[1]
        else:
            weights[indices] = policy[2]
        if normalize_dialogues:
            weights[indices] /= weights[indices].sum()
    return weights / weights.mean()


def fit_binary(x, y, label, config, weights):
    c, weight = config
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = LinearSVC(C=c, class_weight=weight, dual=False, max_iter=5000, random_state=42)
        model.fit(x, y == label, sample_weight=weights)
    scale = max(float(np.std(model.decision_function(x))), 1e-06)
    return (model, scale)


def train_tasks(xt, xv, pairs, configs, policies, n_jobs, normalize_dialogues=False):
    y = pairs.reaction.to_numpy()
    weights = {p: sample_weights(pairs, p, normalize_dialogues) for p in policies}
    tasks = [(p, label, k, config) for p in policies for label in ['positive', 'negative'] for k, config in enumerate(configs)]

    def work(task):
        p, label, k, config = task
        model, scale = fit_binary(xt, y, label, config, weights[p])
        return (p, label, k, model.decision_function(xv) / scale)
    if n_jobs == 1:
        return list(map(work, tasks))
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(work, tasks))


def splits(df, n):
    for tr, va in StratifiedGroupKFold(n, shuffle=True, random_state=42).split(df, df.reaction, df.group):
        assert not set(df.group.iloc[tr]) & set(df.group.iloc[va])
        yield (tr, va)


def select(train, configs, policies, inner_splits, n_jobs, normalize_dialogues=False):
    pairs = g.interactions(train, True)
    all_scores = {p: {label: np.full((len(configs), len(pairs)), np.nan) for label in ['positive', 'negative']} for p in policies}
    for tr, va in splits(train, inner_splits):
        pt = g.interactions(train.iloc[tr], True)
        pv = g.interactions(train.iloc[va].drop(columns=['labels', 'reaction']))
        _, xt, xv = next(j.matrix_variants(pt, pv, ['G_signals']))
        mask = pairs.dialog_pos.isin(va).to_numpy()
        assert mask.sum() == len(pv)
        for p, label, k, values in train_tasks(xt, xv, pt, configs, policies, n_jobs, normalize_dialogues=False):
            all_scores[p][label][k, mask] = values
    selections = {}
    searches = []
    for policy, scores in all_scores.items():
        assert all((np.isfinite(v).all() for v in scores.values()))
        choice, search = j.choose_fast(train, pairs, scores, configs)
        selections[policy] = choice
        search['decisive_weight'] = policy[0]
        search['late_neutral_weight'] = policy[1]
        search['neutral_dialog_weight'] = policy[2]
        searches.append(search)
    best = policies[0]
    for policy in policies[1:]:
        if selections[policy]['inner_macro_f1'] > selections[best]['inner_macro_f1'] + 1e-12:
            best = policy
    return (selections, best, pd.concat(searches, ignore_index=True))


def outer_predictions(train, valid, selections, policies, normalize_dialogues=False):
    pt = g.interactions(train, True)
    pv = g.interactions(valid[['dialog_id', 'turns']])
    _, xt, xv = next(j.matrix_variants(pt, pv, ['G_signals']))
    results = {}
    for policy in dict.fromkeys(policies):
        choice = selections[policy]
        weights = sample_weights(pt, policy, normalize_dialogues)
        scores = {}
        for label in ['positive', 'negative']:
            model, scale = fit_binary(xt, pt.reaction.to_numpy(), label, choice[label], weights)
            scores[label] = model.decision_function(xv) / scale
        pred = g.aggregate(pv, scores['positive'], scores['negative'], len(valid), choice['thresholds'])
        ep = scores['positive'] - choice['thresholds'][0]
        en = scores['negative'] - choice['thresholds'][1]
        turns = pd.DataFrame({'dialog_id': valid.dialog_id.to_numpy()[pv.dialog_pos.to_numpy()], 'user_number': pv.groupby('dialog_pos').cumcount().to_numpy() + 2, 'positive_score': scores['positive'], 'negative_score': scores['negative'], 'positive_threshold': choice['thresholds'][0], 'negative_threshold': choice['thresholds'][1], 'prediction': np.where((ep >= 0) | (en >= 0), np.where(ep > en, 'positive', 'negative'), 'neutral')})
        results[policy] = dict(pred=pred, turn_scores=turns, choice=choice, policy=policy)
    return results


def fingerprint(train, configs, policies, outer, inner, normalize_dialogues=False):
    source = b''.join((Path(m.__file__).read_bytes() for m in [g, j, a])) + Path(__file__).read_bytes()
    data = train[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records')
    return hashlib.sha256(source + data.encode() + repr((configs, policies, outer, inner, normalize_dialogues)).encode()).hexdigest()


def run(
    train,
    configs=g.CONFIGS,
    policies=POLICIES,
    outer_splits=5,
    inner_splits=3,
    n_jobs=2,
    normalize_dialogues=False,
):
    if (1.0, 1.0, 1.0) not in policies:
        raise ValueError('Control weights (1,1,1) required')
    if any((policy not in POLICY_NAMES for policy in policies)):
        raise ValueError('Unknown decision policy')
    if n_jobs not in [1, 2, 3, 4]:
        raise ValueError('n_jobs must be 1..4')
    train = train.reset_index(drop=True)
    key = fingerprint(train, configs, policies, outer_splits, inner_splits, normalize_dialogues)
    dest = RESULT_DIR / key[:16]
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'results.joblib').exists():
        return joblib.load(dest / 'results.joblib')
    predictions = {name: np.full(len(train), '', dtype=object) for name in [POLICY_NAMES[p] for p in policies]}
    folds = []
    settings = []
    turn_tables = []
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(splits(train, outer_splits), 1):
            cache = dest / f'fold{fold}.joblib'
            if cache.exists():
                data = joblib.load(cache)
            else:
                fit = train.iloc[tr].reset_index(drop=True)
                valid = train.iloc[va][['dialog_id', 'turns']].reset_index(drop=True)
                selections, best, search = select(
                    fit,
                    configs,
                    policies,
                    inner_splits,
                    n_jobs,
                    normalize_dialogues,
                )
                search.to_csv(dest / f'fold{fold}_search.csv', index=False)
                outputs = outer_predictions(fit, valid, selections, policies, normalize_dialogues)
                data = {POLICY_NAMES[policy]: outputs[policy] for policy in policies}
                joblib.dump(data, cache)
            for name, values in data.items():
                predictions[name][va] = values['pred']
                score = a.macro(train.reaction.to_numpy()[va], values['pred'])
                folds.append(dict(variant=name, fold=fold, macro_f1=score))
                settings.append(dict(variant=name, fold=fold, policy=values['policy'], **values['choice']))
                turn_tables.append(values['turn_scores'].assign(variant=name, fold=fold))
    assert all(((v != '').all() for v in predictions.values()))
    ft = pd.DataFrame(folds)
    table = pd.DataFrame([dict(variant=name, OOF_Macro_F1=a.macro(train.reaction.to_numpy(), p), Mean_CV_Macro_F1=ft[ft.variant == name].macro_f1.mean()) for name, p in predictions.items()]).set_index('variant')
    table['Delta_vs_G'] = table.OOF_Macro_F1 - table.loc['G_signals', 'OOF_Macro_F1']
    turns = pd.concat(turn_tables, ignore_index=True)
    turns.to_csv(dest / 'oof_turn_scores.csv', index=False)
    result = dict(
        table=table,
        fold_scores=ft,
        settings=settings,
        predictions=predictions,
        y=train.reaction.to_numpy(),
        dialog_ids=train.dialog_id.tolist(),
        fingerprint=key,
        turn_scores=turns,
        normalize_dialogues=normalize_dialogues,
    )
    table.to_csv(dest / 'comparison.csv')
    joblib.dump(result, dest / 'results.joblib')
    return result


def display_result(result):
    from IPython.display import display
    display(result['table'].round(4))
    display(pd.DataFrame(result['settings']))
    paired = result['fold_scores'].pivot(index='fold', columns='variant', values='macro_f1')
    for name in result['table'].index:
        if name != 'G_signals':
            paired[name + '_minus_control'] = paired[name] - paired['G_signals']
    display(paired.round(4))
    if result.get('normalize_dialogues'):
        print('В этом сравнении веса нормированы по диалогам.')
    for name, pred in result['predictions'].items():
        print(name)
        g.display_result(dict(y=result['y'], predictions=pred, macro_f1=result['table'].loc[name, 'OOF_Macro_F1'], mean_cv_macro_f1=result['table'].loc[name, 'Mean_CV_Macro_F1']))
