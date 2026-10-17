from pathlib import Path
import hashlib
import itertools
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from . import course_ovr as g, course_interactions as j, course_ablation as a
RESULT_DIR = Path('artifacts/course_gate')
GATE_C = [0.1, 1.0, 10.0]
GATE_THRESHOLDS = [0.2, 0.35, 0.5, 0.65, 0.8]
NAMES = ['G_signals', 'LR_sequence', 'LR_G_gate']


def splits(df, n):
    for tr, va in StratifiedGroupKFold(n, shuffle=True, random_state=42).split(df, df.reaction, df.group):
        assert not set(df.group.iloc[tr]) & set(df.group.iloc[va])
        yield (tr, va)


def user_text(dialogs):
    return ['\n'.join((t['text'] for t in turns if t['role'] == 'user')) for turns in dialogs.turns]


def text_encoder():
    return FeatureUnion([('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True, dtype=np.float32)), ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=18000, sublinear_tf=True, dtype=np.float32))])


def g_outputs(pairs, sp, sn, n, choice):
    ep = sp - choice['thresholds'][0]
    en = sn - choice['thresholds'][1]
    baseline = g.aggregate(pairs, sp, sn, n, choice['thresholds'])
    ids = pairs.dialog_pos.to_numpy(dtype=int)
    counts = np.bincount(ids, minlength=n)
    if (counts == 0).any():
        raise ValueError('No candidate user turns')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    ends = starts + counts - 1
    confidence = np.maximum(ep, en)
    best = np.maximum.reduceat(confidence, starts)
    loc = np.maximum.reduceat(np.where(confidence == best[ids], np.arange(len(ids)), -1), starts)
    forced = np.where(ep[loc] > en[loc], 'positive', 'negative').astype(object)
    sign = np.where(baseline == 'neutral', forced, baseline)
    numeric = []
    for values in [sp, sn, ep, en]:
        numeric.extend([np.add.reduceat(values, starts) / counts, np.maximum.reduceat(values, starts), np.minimum.reduceat(values, starts), values[ends]])
    top2 = np.array([np.sort(confidence[start:end + 1])[-2:].mean() for start, end in zip(starts, ends)])
    numeric.extend([top2, np.add.reduceat(confidence, starts) / counts, np.add.reduceat((confidence >= 0).astype(float), starts)])
    for back in range(3):
        loc_tail = ends - back
        numeric.append(np.where(loc_tail >= starts, confidence[np.maximum(loc_tail, 0)], 0.0))
    signal = j.InteractionSignals().transform(pairs).toarray()
    names = list(j.InteractionSignals().get_feature_names_out())
    for name in ['assistant_apology', 'following_apology', 'following_admission', 'user_gratitude', 'user_revision']:
        numeric.append(np.add.reduceat(signal[:, names.index(name)], starts))
    numeric.extend([counts, (baseline == 'positive').astype(float), (baseline == 'negative').astype(float), best, (loc - starts) / np.maximum(counts - 1, 1)])
    return (baseline, sign, np.column_stack(numeric).astype(np.float32))


def fitted_outputs(bundle, dialogs):
    pairs = g.interactions(dialogs[['dialog_id', 'turns']].reset_index(drop=True))
    x = bundle['encoder'].transform(pairs)
    s = {label: model.decision_function(x) / scale for label, (model, scale) in bundle['models'].items()}
    return g_outputs(pairs, s['positive'], s['negative'], len(dialogs), bundle['choice'])


def prepare_g(train, valid, configs, n_jobs, sub_splits):
    source = b''.join((Path(m.__file__).read_bytes() for m in [g, j, a])) + Path(__file__).read_bytes()
    train_data = train[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records')
    valid_data = valid[['dialog_id', 'turns']].to_json(orient='records')
    key = hashlib.sha256(source + train_data.encode() + valid_data.encode() + repr((configs, sub_splits)).encode()).hexdigest()
    cache = RESULT_DIR / 'g_score_cache' / f'{key}.joblib'
    if cache.exists():
        return joblib.load(cache)
    pairs, scores = j.inner_scores_shared(train, ['G_signals'], configs, sub_splits, n_jobs)
    scores = scores['G_signals']
    choice, _ = j.choose_fast(train, pairs, scores, configs)
    kp = configs.index(choice['positive'])
    kn = configs.index(choice['negative'])
    train_out = g_outputs(pairs, scores['positive'][kp], scores['negative'][kn], len(train), choice)
    bundle = j.fit(train, 'G_signals', choice)
    outputs = (train_out, fitted_outputs(bundle, valid))
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(outputs, cache)
    return outputs


def gate_probabilities(train, valid, train_numeric, valid_numeric):
    enc = text_encoder()
    xt = enc.fit_transform(user_text(train))
    xv = enc.transform(user_text(valid))
    scale = StandardScaler()
    nt = scale.fit_transform(train_numeric)
    nv = scale.transform(valid_numeric)
    target = (train.reaction.to_numpy() != 'neutral').astype(int)
    out = {name: [] for name in NAMES[1:]}
    for name in out:
        ft, fv = (
            nt,
            nv,
        ) if name == 'LR_sequence' else (sparse.hstack([xt, nt], format='csr'), sparse.hstack([xv, nv], format='csr'))
        for c in GATE_C:
            model = LogisticRegression(
                C=c,
                class_weight='balanced',
                solver='liblinear',
                max_iter=2000,
                random_state=42,
            )
            model.fit(ft, target)
            if model.n_iter_.max() >= model.max_iter:
                raise RuntimeError('LR did not converge')
            out[name].append(model.predict_proba(fv)[:, list(model.classes_).index(1)])
    return {name: np.asarray(values) for name, values in out.items()}


def choose_gate(y, probabilities, sign):
    best = None
    rows = []
    for k, c in enumerate(GATE_C):
        for threshold in GATE_THRESHOLDS:
            pred = np.where(probabilities[k] >= threshold, sign, 'neutral')
            value = a.macro(y, pred)
            rows.append(dict(C=c, threshold=threshold, macro_f1=value))
            if best is None or value > best['macro_f1'] + 1e-12:
                best = dict(index=k, C=c, threshold=threshold, macro_f1=value)
    return (best, pd.DataFrame(rows))


def fingerprint(training, configs, outer_splits, inner_splits, sub_splits):
    source = b''.join((Path(m.__file__).read_bytes() for m in [g, j, a])) + Path(__file__).read_bytes()
    data = training[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records')
    return hashlib.sha256(source + data.encode() + repr((configs, outer_splits, inner_splits, sub_splits)).encode()).hexdigest()


def run(training, configs=g.CONFIGS, outer_splits=5, inner_splits=3, sub_splits=2, n_jobs=2):
    training = training.reset_index(drop=True)
    key = fingerprint(training, configs, outer_splits, inner_splits, sub_splits)
    dest = RESULT_DIR / key[:16]
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'results.joblib').exists():
        return joblib.load(dest / 'results.joblib')
    predictions = {name: np.full(len(training), '', dtype=object) for name in NAMES}
    folds = []
    settings = []
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(splits(training, outer_splits), 1):
            fd = dest / f'fold_{fold}'
            fd.mkdir(exist_ok=True)
            cached = fd / 'predictions.joblib'
            if cached.exists():
                data = joblib.load(cached)
            else:
                train = training.iloc[tr].reset_index(drop=True)
                valid = training.iloc[va].drop(columns=['labels', 'reaction']).reset_index(drop=True)
                inner_prob = {name: np.full((len(GATE_C), len(train)), np.nan) for name in NAMES[1:]}
                inner_sign = np.full(len(train), '', dtype=object)
                for part, (it, iv) in enumerate(splits(train, inner_splits), 1):
                    pc = fd / f'inner_{part}.joblib'
                    if pc.exists():
                        part_result = joblib.load(pc)
                    else:
                        fit = train.iloc[it].reset_index(drop=True)
                        val = train.iloc[iv].drop(columns=['labels', 'reaction']).reset_index(drop=True)
                        fit_out, val_out = prepare_g(fit, val, configs, n_jobs, sub_splits)
                        probabilities = gate_probabilities(fit, val, fit_out[2], val_out[2])
                        part_result = dict(probabilities=probabilities, sign=val_out[1])
                        joblib.dump(part_result, pc)
                    inner_sign[iv] = part_result['sign']
                    for name in inner_prob:
                        inner_prob[name][:, iv] = part_result['probabilities'][name]
                chosen = {}
                for name in inner_prob:
                    assert np.isfinite(inner_prob[name]).all()
                    choice, search = choose_gate(train.reaction.to_numpy(), inner_prob[name], inner_sign)
                    chosen[name] = choice
                    search.to_csv(fd / f'{name}_search.csv', index=False)
                train_out, val_out = prepare_g(train, valid, configs, n_jobs, inner_splits)
                probabilities = gate_probabilities(train, valid, train_out[2], val_out[2])
                pred = {'G_signals': val_out[0]}
                for name, choice in chosen.items():
                    pred[name] = np.where(probabilities[name][choice['index']] >= choice['threshold'], val_out[1], 'neutral')
                data = dict(predictions=pred, chosen=chosen)
                joblib.dump(data, cached)
            for name, pred in data['predictions'].items():
                predictions[name][va] = pred
                score = a.macro(training.reaction.to_numpy()[va], pred)
                folds.append(dict(variant=name, fold=fold, macro_f1=score))
            settings.append(dict(fold=fold, **data['chosen']))
    assert all(((v != '').all() for v in predictions.values()))
    ft = pd.DataFrame(folds)
    table = pd.DataFrame([dict(variant=name, OOF_Macro_F1=a.macro(training.reaction.to_numpy(), p), Mean_CV_Macro_F1=ft[ft.variant == name].macro_f1.mean()) for name, p in predictions.items()]).set_index('variant')
    table['Delta_vs_G_signals'] = table.OOF_Macro_F1 - table.loc['G_signals', 'OOF_Macro_F1']
    result = dict(
        table=table,
        fold_scores=ft,
        settings=settings,
        predictions=predictions,
        y=training.reaction.to_numpy(),
        dialog_ids=training.dialog_id.tolist(),
        fingerprint=key,
    )
    table.to_csv(dest / 'comparison.csv')
    joblib.dump(result, dest / 'results.joblib')
    return result
