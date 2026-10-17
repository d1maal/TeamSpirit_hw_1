from pathlib import Path
import hashlib
import copy
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from . import course_ovr as g, course_interactions as j, course_ablation as a, course_decision_weight as d, course_positive_weight as p
RESULT_DIR = Path('artifacts/course_nb_gate')
CONFIGS = p.CONFIGS
GATE_C = [0.1, 1.0, 10.0]
GATE_THRESHOLDS = [0.2, 0.35, 0.5, 0.65, 0.8]
NAMES = ['Decision_2x', 'Gate_G', 'Gate_G_NB']


def role_encoder(encoder):
    base = copy.deepcopy(dict(encoder.transformer_list)['G'])
    base.transformers_ = [(name, t, cols) for name, t, cols in base.transformers_ if name not in ['structure', 'remainder']]
    return base


def nb_scores(encoder, pt, pv):
    enc = role_encoder(encoder)
    xt = enc.transform(pt)
    xv = enc.transform(pv)
    if xt.nnz and xt.data.min() < 0:
        raise ValueError('ComplementNB requires nonnegative features')
    nb = ComplementNB(alpha=1.0, norm=False)
    nb.fit(xt, (pt.reaction.to_numpy() != 'neutral').astype(int))
    logp = nb.predict_log_proba(xv)
    classes = list(nb.classes_)
    return (logp[:, classes.index(1)] - logp[:, classes.index(0)], nb)


def aggregate(pairs, sp, sn, nb_score, choice, n):
    ids = pairs.dialog_pos.to_numpy(dtype=int)
    counts = np.bincount(ids, minlength=n)
    if (counts == 0).any():
        raise ValueError('No candidate reactions')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    ends = starts + counts
    ep = sp - choice['thresholds'][0]
    en = sn - choice['thresholds'][1]
    turnpred = np.where((ep >= 0) | (en >= 0), np.where(ep > en, 'positive', 'negative'), 'neutral')
    raw = g.aggregate(pairs, sp, sn, n, choice['thresholds'])
    signals = j.InteractionSignals().transform(pairs).toarray()
    features = []
    nbfeatures = []
    sign = []
    for start, end, baseline in zip(starts, ends, raw):
        epos, eneg = (ep[start:end], en[start:end])
        confidence = np.maximum(epos, eneg)
        found = np.flatnonzero(turnpred[start:end] != 'neutral')
        selected = found[-1] if len(found) else np.flatnonzero(confidence == confidence.max())[-1]
        sign.append('positive' if epos[selected] > eneg[selected] else 'negative')
        row = []
        for values in [sp[start:end], sn[start:end], confidence]:
            row.extend([values.mean(), values.max(), values.std(), values[-1], np.sort(values)[-2:].mean()])
        row.extend([(confidence >= 0).sum(), len(confidence)])
        row.extend(signals[start:end].mean(axis=0))
        row.extend(signals[start:end].max(axis=0))
        features.append(row)
        values = nb_score[start:end]
        ordered = np.sort(values)
        nbfeatures.append([values.max(), values.mean(), values.std(), values[-1], ordered[-2:].mean(), ordered[-3:].mean(), values.sum(), (values > 0).sum()])
    return dict(baseline=raw, sign=np.array(sign, dtype=object), G=np.asarray(features, dtype=np.float32), NB=np.asarray(nbfeatures, dtype=np.float32))


def prepare(train, valid, configs, inner_splits, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_cache = cache_dir / ('gate_inputs_' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12] + '.joblib')
    if output_cache.exists():
        return joblib.load(output_cache)
    choices, _ = p.select(train, ['Decision_2x'], configs, inner_splits, cache_dir)
    choice = choices['Decision_2x']
    kp = configs.index(choice['positive'])
    kn = configs.index(choice['negative'])
    pairs = g.interactions(train, True)
    sp = np.full(len(pairs), np.nan)
    sn = sp.copy()
    nb = sp.copy()
    for part, (tr, va) in enumerate(d.splits(train, inner_splits), 1):
        saved = joblib.load(cache_dir / f'inner{part}_models.joblib')
        pt = g.interactions(train.iloc[tr], True)
        pv = g.interactions(train.iloc[va][['dialog_id', 'turns']])
        mask = pairs.dialog_pos.isin(va).to_numpy()
        sp[mask] = saved['positive']['Decision_2x'][kp]
        sn[mask] = saved['negative'][kn]
        nb[mask], model = nb_scores(saved['encoder'], pt, pv)
        joblib.dump(model, cache_dir / f'inner{part}_nb.joblib')
    assert np.isfinite(sp).all() and np.isfinite(sn).all() and np.isfinite(nb).all()
    training = aggregate(pairs, sp, sn, nb, choice, len(train))
    out = p.outer(train, valid, choices, cache_dir)['Decision_2x']
    saved = joblib.load(cache_dir / 'outer_models.joblib')
    pt = g.interactions(train, True)
    pv = g.interactions(valid[['dialog_id', 'turns']])
    val_nb, model = nb_scores(saved['encoder'], pt, pv)
    joblib.dump(model, cache_dir / 'outer_nb.joblib')
    validation = aggregate(
        pv,
        out['turn_scores'].positive_score.to_numpy(),
        out['turn_scores'].negative_score.to_numpy(),
        val_nb,
        choice,
        len(valid),
    )
    value = (training, validation)
    joblib.dump(value, output_cache)
    return value


def probabilities(train_inputs, valid_inputs, y):
    out = {}
    for name in NAMES[1:]:
        xt = train_inputs['G']
        xv = valid_inputs['G']
        if name == 'Gate_G_NB':
            xt = np.column_stack([xt, train_inputs['NB']])
            xv = np.column_stack([xv, valid_inputs['NB']])
        scaler = StandardScaler()
        xt = scaler.fit_transform(xt)
        xv = scaler.transform(xv)
        values = []
        for c in GATE_C:
            model = LogisticRegression(
                C=c,
                class_weight='balanced',
                solver='liblinear',
                max_iter=2000,
                random_state=42,
            ).fit(xt, y)
            if model.n_iter_.max() >= model.max_iter:
                raise RuntimeError('Gate LR did not converge')
            values.append(model.predict_proba(xv)[:, list(model.classes_).index(1)])
        out[name] = np.array(values)
    return out


def fingerprint(train, configs, outer, inner, sub):
    source = Path(__file__).read_bytes() + b''.join((Path(m.__file__).read_bytes() for m in [g, j, a, d, p]))
    return hashlib.sha256(source + train[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records').encode() + repr((configs, outer, inner, sub)).encode()).hexdigest()


def run(train, configs=CONFIGS, outer_splits=5, inner_splits=3, sub_splits=2, reuse_p_dir=None):
    train = train.reset_index(drop=True)
    key = fingerprint(train, configs, outer_splits, inner_splits, sub_splits)
    dest = RESULT_DIR / key[:16]
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'results.joblib').exists():
        return joblib.load(dest / 'results.joblib')
    reuse_root = Path(reuse_p_dir) if reuse_p_dir is not None else p.RESULT_DIR
    compatible = reuse_root / p.fingerprint(
        train,
        configs,
        outer_splits,
        inner_splits,
        False,
    )[:16] / 'shared_models'
    predictions = {name: np.full(len(train), '', dtype=object) for name in NAMES}
    folds = []
    settings = []
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(d.splits(train, outer_splits), 1):
            fd = dest / f'fold{fold}'
            fd.mkdir(exist_ok=True)
            cache = fd / 'results.joblib'
            if cache.exists():
                data = joblib.load(cache)
            else:
                fit = train.iloc[tr].reset_index(drop=True)
                valid = train.iloc[va][['dialog_id', 'turns']].reset_index(drop=True)
                prob = {name: np.full((len(GATE_C), len(fit)), np.nan) for name in NAMES[1:]}
                sign = np.full(len(fit), '', dtype=object)
                for part, (it, iv) in enumerate(d.splits(fit, inner_splits), 1):
                    pc = fd / f'gate_inner{part}.joblib'
                    if pc.exists():
                        values = joblib.load(pc)
                    else:
                        sub = fit.iloc[it].reset_index(drop=True)
                        val = fit.iloc[iv][['dialog_id', 'turns']].reset_index(drop=True)
                        ft, fv = prepare(sub, val, configs, sub_splits, fd / f'submodels{part}')
                        values = dict(
                            prob=probabilities(ft, fv, (sub.reaction.to_numpy() != 'neutral').astype(int)),
                            sign=fv['sign'],
                        )
                        joblib.dump(values, pc)
                    for name in prob:
                        prob[name][:, iv] = values['prob'][name]
                    sign[iv] = values['sign']
                chosen = {}
                for name in prob:
                    assert np.isfinite(prob[name]).all()
                    best = None
                    for k, c in enumerate(GATE_C):
                        for threshold in GATE_THRESHOLDS:
                            pred = np.where(prob[name][k] >= threshold, sign, 'neutral')
                            score = a.macro(fit.reaction.to_numpy(), pred)
                            if best is None or score > best['inner_macro_f1'] + 1e-12:
                                best = dict(index=k, C=c, threshold=threshold, inner_macro_f1=score)
                    chosen[name] = best
                shared = compatible / f'fold{fold}'
                if not shared.exists():
                    shared = fd / 'outer_shared'
                ft, fv = prepare(fit, valid, configs, inner_splits, shared)
                finalprob = probabilities(ft, fv, (fit.reaction.to_numpy() != 'neutral').astype(int))
                preds = {'Decision_2x': fv['baseline']}
                for name, choice in chosen.items():
                    preds[name] = np.where(finalprob[name][choice['index']] >= choice['threshold'], fv['sign'], 'neutral')
                data = dict(predictions=preds, settings=chosen)
                joblib.dump(data, cache)
            for name, pred in data['predictions'].items():
                predictions[name][va] = pred
                score = a.macro(train.reaction.to_numpy()[va], pred)
                folds.append(dict(variant=name, fold=fold, macro_f1=score))
            settings.append(dict(fold=fold, **data['settings']))
    assert all(((v != '').all() for v in predictions.values()))
    ft = pd.DataFrame(folds)
    table = pd.DataFrame([dict(variant=name, OOF_Macro_F1=a.macro(train.reaction.to_numpy(), pred), Mean_CV_Macro_F1=ft[ft.variant == name].macro_f1.mean()) for name, pred in predictions.items()]).set_index('variant')
    table['Delta_vs_Decision_2x'] = table.OOF_Macro_F1 - table.loc['Decision_2x', 'OOF_Macro_F1']
    result = dict(
        table=table,
        predictions=predictions,
        y=train.reaction.to_numpy(),
        fold_scores=ft,
        settings=settings,
        dialog_ids=train.dialog_id.tolist(),
        fingerprint=key,
    )
    table.to_csv(dest / 'comparison.csv')
    joblib.dump(result, dest / 'results.joblib')
    return result


def display_result(result):
    from IPython.display import display
    display(result['table'].round(4))
    paired = result['fold_scores'].pivot(index='fold', columns='variant', values='macro_f1')
    paired['NB_minus_G_gate'] = paired.Gate_G_NB - paired.Gate_G
    display(paired.round(4))
    for name, pred in result['predictions'].items():
        print(name)
        g.display_result(dict(y=result['y'], predictions=pred, macro_f1=result['table'].loc[name, 'OOF_Macro_F1'], mean_cv_macro_f1=result['table'].loc[name, 'Mean_CV_Macro_F1']))
