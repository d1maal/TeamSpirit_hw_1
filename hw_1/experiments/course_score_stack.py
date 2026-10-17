from pathlib import Path
import hashlib
import sys
import types
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import warnings
from . import course_ovr as g, course_interactions as j, course_nb_gate as q, course_positive_weight as p, course_decision_weight as d
RESULT_DIR = Path('artifacts/course_score_stack')
FEATURE_NAMES = [
    'positive_max',
    'negative_max',
    'nonneutral_max',
    'nonneutral_top2',
    'last_detected_nonneutral_margin',
    'has_detected_nonneutral',
    'NB_log_neutral_at_max',
    'NB_log_nonneutral_at_max',
    'NB_max_log_odds',
    'user_gratitude',
    'user_disagreement',
    'following_apology',
]


def load_weights(path):
    aliases = []
    for prefix in ['positive_g_runtime', 'nb_gate_runtime']:
        for name, module in [(prefix, None), (prefix + '.course_interactions', j)]:
            if name not in sys.modules:
                if module is None:
                    module = types.ModuleType(prefix)
                    module.__path__ = []
                sys.modules[name] = module
                aliases.append(name)
    try:
        return joblib.load(path)
    finally:
        for name in reversed(aliases):
            sys.modules.pop(name, None)


def last_margin(ids, sp, sn, thresholds, n):
    ep = sp - thresholds[0]
    en = sn - thresholds[1]
    margin = np.maximum(ep, en)
    last = np.full(n, -1, dtype=int)
    np.maximum.at(last, ids, np.where(margin >= 0, np.arange(len(ids)), -1))
    out = np.zeros(n)
    found = last >= 0
    out[found] = margin[last[found]]
    return out


def small_features(cached, last):
    G = cached['G']
    NB = cached['NB']
    if G.shape[1] != 77 or NB.shape[1] != 8:
        raise ValueError('Unexpected Q feature schema')
    odds = NB[:, 0].astype(float)
    signal_names = list(j.InteractionSignals().get_feature_names_out())
    columns = [
        G[:, 1],
        G[:, 6],
        G[:, 11],
        G[:, 14],
        last,
        (G[:, 15] > 0).astype(float),
        -np.logaddexp(0.0, odds),
        -np.logaddexp(0.0, -odds),
        odds,
    ]
    for name in ['user_gratitude', 'user_disagreement', 'following_apology']:
        columns.append(G[:, 47 + signal_names.index(name)])
    x = np.column_stack(columns)
    if x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid meta features')
    return x


def fold_features(fit, valid, cache_dir, configs, inner_splits):
    name = 'gate_inputs_' + hashlib.sha256(Path(q.__file__).read_bytes()).hexdigest()[:12] + '.joblib'
    path = cache_dir / name
    if not path.exists():
        raise FileNotFoundError(f'Need compatible saved Q scores: {path}; base training is disabled')
    training, validation = joblib.load(path)
    saved = load_weights(cache_dir / 'outer_models.joblib')
    outer = saved['outputs']['Decision_2x']
    choice = outer['choice']
    kp = configs.index(choice['positive'])
    kn = configs.index(choice['negative'])
    pairs = g.interactions(fit, True)
    sp = np.full(len(pairs), np.nan)
    sn = sp.copy()
    for part, (tr, va) in enumerate(d.splits(fit, inner_splits), 1):
        saved = load_weights(cache_dir / f'inner{part}_models.joblib')
        mask = pairs.dialog_pos.isin(va).to_numpy()
        sp[mask] = saved['positive']['Decision_2x'][kp]
        sn[mask] = saved['negative'][kn]
    if not np.isfinite(sp).all() or not np.isfinite(sn).all():
        raise ValueError('Incomplete inner OOF')
    trainlast = last_margin(pairs.dialog_pos.to_numpy(dtype=int), sp, sn, choice['thresholds'], len(fit))
    turn = outer['turn_scores']
    pv = g.interactions(valid[['dialog_id', 'turns']])
    expected = valid.dialog_id.to_numpy()[pv.dialog_pos.to_numpy()]
    if not np.array_equal(turn.dialog_id.to_numpy(), expected):
        raise ValueError('Cached validation turn order differs')
    vallast = last_margin(
        pv.dialog_pos.to_numpy(dtype=int),
        turn.positive_score.to_numpy(),
        turn.negative_score.to_numpy(),
        choice['thresholds'],
        len(valid),
    )
    if not np.array_equal(validation['baseline'], outer['pred']):
        raise ValueError('Cached baseline predictions disagree')
    return (small_features(training, trainlast), small_features(validation, vallast), validation['baseline'])


def run(train, q_dir, reuse_p_dir=None, min_delta=0.01):
    train = train.reset_index(drop=True)
    q_dir = Path(q_dir)
    saved = joblib.load(q_dir / 'results.joblib')
    if saved.get('fingerprint') != q.fingerprint(train, q.CONFIGS, 5, 3, 2):
        raise ValueError('Q code/data/configuration fingerprint differs; cannot safely reuse cached features')
    if saved['dialog_ids'] != train.dialog_id.tolist() or not np.array_equal(saved['y'], train.reaction):
        raise ValueError('Q labels or IDs differ')
    p_root = Path(reuse_p_dir) if reuse_p_dir is not None else p.RESULT_DIR
    p_shared = p_root / p.fingerprint(train, p.CONFIGS, 5, 3, False)[:16] / 'shared_models'
    key = hashlib.sha256(Path(__file__).read_bytes() + saved['fingerprint'].encode()).hexdigest()
    dest = RESULT_DIR / key[:16]
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'results.joblib').exists():
        return joblib.load(dest / 'results.joblib')
    predictions = {
        'Decision_2x': np.asarray(saved['predictions']['Decision_2x']),
        'Score_LR': np.full(len(train), '', dtype=object),
    }
    folds = []
    coefs = []
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(d.splits(train, 5), 1):
            cache_dir = p_shared / f'fold{fold}'
            if not cache_dir.exists():
                cache_dir = q_dir / f'fold{fold}/outer_shared'
            fit = train.iloc[tr].reset_index(drop=True)
            valid = train.iloc[va][['dialog_id', 'turns']].reset_index(drop=True)
            xt, xv, baseline = fold_features(fit, valid, cache_dir, q.CONFIGS, 3)
            if not np.array_equal(baseline, predictions['Decision_2x'][va]):
                raise ValueError('Fold baseline does not match Q')
            scaler = StandardScaler()
            xt = scaler.fit_transform(xt)
            xv = scaler.transform(xv)
            model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                solver='lbfgs',
                max_iter=2000,
                random_state=42,
            )
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(xt, fit.reaction.to_numpy())
            predictions['Score_LR'][va] = model.predict(xv)
            joblib.dump(
                dict(scaler=scaler, model=model, feature_names=FEATURE_NAMES, train_ids=fit.dialog_id.tolist(), validation_ids=valid.dialog_id.tolist()),
                dest / f'meta_fold{fold}.joblib',
            )
            for label, coef in zip(model.classes_, model.coef_):
                coefs.extend((dict(fold=fold, label=label, feature=name, coefficient=value) for name, value in zip(FEATURE_NAMES, coef)))
            for name, pred in predictions.items():
                value = f1_score(
                    train.reaction.to_numpy()[va],
                    pred[va],
                    labels=g.CLASSES,
                    average='macro',
                    zero_division=0,
                )
                folds.append(dict(variant=name, fold=fold, macro_f1=value))
    ft = pd.DataFrame(folds)
    table = pd.DataFrame([dict(variant=name, OOF_Macro_F1=f1_score(train.reaction, pred, labels=g.CLASSES, average='macro', zero_division=0), Mean_CV_Macro_F1=ft[ft.variant == name].macro_f1.mean()) for name, pred in predictions.items()]).set_index('variant')
    table['Delta_vs_Decision_2x'] = table.Mean_CV_Macro_F1 - table.loc['Decision_2x', 'Mean_CV_Macro_F1']
    delta = float(table.loc['Score_LR', 'Delta_vs_Decision_2x'])
    result = dict(
        table=table,
        predictions=predictions,
        y=train.reaction.to_numpy(),
        fold_scores=ft,
        coefficients=pd.DataFrame(coefs),
        dialog_ids=train.dialog_id.tolist(),
        delta=delta,
        feature_names=FEATURE_NAMES,
        fingerprint=key,
    )
    table.to_csv(dest / 'comparison.csv')
    result['coefficients'].to_csv(dest / 'coefficients.csv', index=False)
    pd.DataFrame({'dialog_id': train.dialog_id, 'truth': train.reaction, **predictions}).to_csv(dest / 'oof_predictions.csv', index=False)
    joblib.dump(result, dest / 'results.joblib')
    return result


def display_result(result, min_delta=0.01):
    from IPython.display import display
    display(result['table'].round(4))
    paired = result['fold_scores'].pivot(index='fold', columns='variant', values='macro_f1')
    paired['delta'] = paired.Score_LR - paired.Decision_2x
    display(paired.round(4))
    print(f"Изменение среднего Macro-F1: {result['delta']:+.4f}; минимальный прирост={min_delta:.3f}")
    print('Нужный прирост есть.' if result['delta'] >= min_delta else 'Нужного прироста нет.')
    for name, pred in result['predictions'].items():
        print(name)
        g.display_result(dict(y=result['y'], predictions=pred, macro_f1=result['table'].loc[name, 'OOF_Macro_F1'], mean_cv_macro_f1=result['table'].loc[name, 'Mean_CV_Macro_F1']))
