from pathlib import Path
import hashlib
import re
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from . import course_ovr as g, course_interactions as j
RESULT_DIR = Path('artifacts/binary_feedback')
GATE_C = [0.1, 1.0, 10.0]
THRESHOLDS = [0.2, 0.35, 0.5, 0.65, 0.8]
CUES = {
    'thanks': r'\b(?:thanks|thank you|appreciate|cheers|grateful)\b',
    'complaint': r"\b(?:wrong|incorrect|terrible|unhelpful|doesn.t work|you missed|not what i (?:asked|requested)|still haven.t|too vague)\b",
    'revision': r'\b(?:fix|redo|revise|rewrite|try again|instead|correct (?:it|this|that))\b',
    'success': r"\b(?:it works|exactly|that.s it|you nailed it|perfect|great|helpful)\b",
    'detail': r"\b(?:more detail|didn.t explain|not enough|too vague|explain)\b",
    'continuation': r'\b(?:go on|continue|what about|next|another example)\b',
    'admission': r"\b(?:you(?:.re| are) right|my mistake|i made (?:a |an )?(?:mistake|error)|i misunderstood|i was wrong)\b",
    'apology': r'\b(?:sorry|apologize|apologise|apologies)\b',
    'repair': r"\b(?:let me correct|i.ll revise|let me fix|corrected|revised)\b",
    'welcome': r"\b(?:you(?:.re| are) welcome|glad i could help|happy to help)\b",
    'ordinary': r"\b(?:certainly|sure|here.s|here is)\b",
    'negation': r"\b(?:not|no|never|don.t|doesn.t|didn.t|haven.t|isn.t)\b",
}


def split(df, n):
    for tr, va in StratifiedGroupKFold(n, shuffle=True, random_state=42).split(df, df.reaction, df.group):
        assert not set(df.group.iloc[tr]) & set(df.group.iloc[va])
        yield (tr, va)


def pairs(dialogs, training=False):
    p = g.interactions(dialogs, training)
    heads = []
    tails = []
    for row in dialogs.to_dict('records'):
        turns = row['turns']
        users = [i for i, t in enumerate(turns) if t['role'] == 'user']
        for i in users[1:]:
            text = turns[i + 1]['text'] if i + 1 < len(turns) and turns[i + 1]['role'] == 'assistant' else ''
            sentences = list(re.finditer(r'[.!?](?:\s+|$)', text[:600]))
            cut = sentences[min(1, len(sentences) - 1)].end() if sentences else min(300, len(text))
            heads.append(text[:cut])
            tails.append(text[cut:][:1200])
    p['head'] = heads
    p['tail'] = tails
    return p


def feedback(p):
    cols = []
    names = []
    for source in ['user', 'head', 'tail']:
        texts = p[source].str.lower()
        for name, pattern in CUES.items():
            cols.append(texts.str.contains(pattern, regex=True).to_numpy(dtype=float))
            names.append(source + '_' + name)
    values = np.column_stack(cols)
    lookup = {name: values[:, i] for i, name in enumerate(names)}
    for name, left, right in [('repair_agreement', 'user_complaint', 'head_admission'), ('repair_apology', 'user_revision', 'head_apology'), ('positive_completion', 'user_success', 'head_welcome'), ('thanks_response', 'user_thanks', 'head_welcome'), ('ordinary_continuation', 'user_continuation', 'head_ordinary'), ('negative_detail', 'user_detail', 'user_negation')]:
        cols.append(lookup[left] * lookup[right])
        names.append(name)

    def overlap(u, v):
        a = set(re.findall(r'\w+', u.lower()))
        b = set(re.findall(r'\w+', v.lower()))
        return len(a & b) / max(len(a | b), 1)
    cols.append(np.array([overlap(u, v) for u, v in zip(p.user, p.assistant)]) * lookup['user_negation'])
    names.append('negated_previous_overlap')
    return (np.column_stack(cols).astype(np.float32), names)


def aggregate(p, probabilities, n):
    ids = p.dialog_pos.to_numpy(dtype=int)
    counts = np.bincount(ids, minlength=n)
    if (counts == 0).any():
        raise ValueError('Dialogue without candidate reactions')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    ends = starts + counts
    cues, names = feedback(p)
    rows = []
    for start, end in zip(starts, ends):
        prob = probabilities[start:end]
        ordered = np.sort(prob)
        row = [
            prob.max(),
            ordered[-2:].mean(),
            ordered[-3:].mean(),
            prob.sum(),
            prob.mean(),
            prob.std(),
            prob[-3:].max(),
        ]
        row.extend([(prob > threshold).sum() for threshold in [0.35, 0.5, 0.65]])
        row.extend([prob[-k] if len(prob) >= k else 0.0 for k in [1, 2, 3]])
        for matrix in [cues[start:end]]:
            row.extend(matrix.max(axis=0))
            row.extend(matrix.sum(axis=0))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def context_model():
    return j.encoder('G_signals')


def fit_context(train):
    p = pairs(train, True)
    enc = context_model()
    x = enc.fit_transform(p)
    model = LogisticRegression(
        C=0.3,
        class_weight='balanced',
        solver='liblinear',
        max_iter=2000,
        random_state=42,
    )
    model.fit(x, (p.reaction.to_numpy() != 'neutral').astype(int))
    if model.n_iter_.max() >= 2000:
        raise RuntimeError('Context LR did not converge')
    return (enc, model)


def predict_context(bundle, valid):
    p = pairs(valid[['dialog_id', 'turns']])
    enc, model = bundle
    probability = model.predict_proba(enc.transform(p))[:, 1]
    return (p, probability)


def train_features(train, folds):
    p = pairs(train, True)
    prob = np.full(len(p), np.nan)
    for tr, va in split(train, folds):
        bundle = fit_context(train.iloc[tr])
        pv, v = predict_context(bundle, train.iloc[va])
        mask = p.dialog_pos.isin(va).to_numpy()
        assert mask.sum() == len(pv)
        prob[mask] = v
    assert np.isfinite(prob).all()
    return aggregate(p, prob, len(train))


def gate_prob(xt, xv, y):
    scale = StandardScaler()
    xt = scale.fit_transform(xt)
    xv = scale.transform(xv)
    out = []
    for c in GATE_C:
        model = LogisticRegression(
            C=c,
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000,
            random_state=42,
        ).fit(xt, y)
        out.append(model.predict_proba(xv)[:, 1])
    return np.array(out)


def metrics(y, p):
    from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, precision_score
    return dict(binary_macro_f1=f1_score(y, p, average='macro', zero_division=0), balanced_accuracy=balanced_accuracy_score(y, p), recall_nonneutral=recall_score(y, p, zero_division=0), recall_neutral=recall_score(y, p, pos_label=0, zero_division=0), precision_nonneutral=precision_score(y, p, zero_division=0))


def fingerprint(train, outer, inner, sub):
    sources = b''.join((Path(m.__file__).read_bytes() for m in [g, j])) + Path(__file__).read_bytes()
    return hashlib.sha256(sources + train[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records').encode() + repr((outer, inner, sub)).encode()).hexdigest()


def run(train, outer_splits=5, inner_splits=3, sub_splits=2):
    train = train.reset_index(drop=True)
    key = fingerprint(train, outer_splits, inner_splits, sub_splits)
    dest = RESULT_DIR / key[:16]
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'results.joblib').exists():
        return joblib.load(dest / 'results.joblib')
    y = (train.reaction.to_numpy() != 'neutral').astype(int)
    pred = np.full(len(train), -1)
    prob = np.full(len(train), np.nan)
    settings = []
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(split(train, outer_splits), 1):
            fd = dest / f'fold{fold}'
            fd.mkdir(exist_ok=True)
            cache = fd / 'result.joblib'
            if cache.exists():
                out = joblib.load(cache)
            else:
                fit = train.iloc[tr].reset_index(drop=True)
                valid = train.iloc[va].drop(columns=['labels', 'reaction'])
                inner = np.full((len(GATE_C), len(fit)), np.nan)
                for part, (it, iv) in enumerate(split(fit, inner_splits), 1):
                    pc = fd / f'inner{part}.joblib'
                    if pc.exists():
                        values = joblib.load(pc)
                    else:
                        ft = fit.iloc[it].reset_index(drop=True)
                        fv = fit.iloc[iv].drop(columns=['labels', 'reaction'])
                        xt = train_features(ft, sub_splits)
                        pv, v = predict_context(fit_context(ft), fv)
                        xv = aggregate(pv, v, len(fv))
                        values = gate_prob(xt, xv, (ft.reaction.to_numpy() != 'neutral').astype(int))
                        joblib.dump(values, pc)
                    inner[:, iv] = values
                best = None
                for k, c in enumerate(GATE_C):
                    for t in THRESHOLDS:
                        score = metrics(
                            (fit.reaction.to_numpy() != 'neutral').astype(int),
                            inner[k] >= t,
                        )['binary_macro_f1']
                        if best is None or score > best[0] + 1e-12:
                            best = (score, k, c, t)
                xt = train_features(fit, inner_splits)
                pv, v = predict_context(fit_context(fit), valid)
                xv = aggregate(pv, v, len(valid))
                values = gate_prob(xt, xv, (fit.reaction.to_numpy() != 'neutral').astype(int))[best[1]]
                out = dict(probability=values, prediction=(values >= best[3]).astype(int), choice=best)
                joblib.dump(out, cache)
            pred[va] = out['prediction']
            prob[va] = out['probability']
            settings.append(out['choice'])
    assert (pred >= 0).all() and np.isfinite(prob).all()
    result = dict(
        y=y,
        predictions=pred,
        probabilities=prob,
        metrics=metrics(y, pred),
        roc_auc=roc_auc_score(y, prob),
        average_precision=average_precision_score(y, prob),
        settings=settings,
        dialog_ids=train.dialog_id.tolist(),
        fingerprint=key,
    )
    joblib.dump(result, dest / 'results.joblib')
    return result


def display_result(result, baseline=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    print(pd.Series(result['metrics']).round(4))
    print(classification_report(result['y'], result['predictions'], target_names=['neutral', 'non-neutral'], zero_division=0))
    if baseline is not None:
        if baseline['dialog_ids'] != result['dialog_ids']:
            raise ValueError('Baseline dialogue order differs')
        b = (np.asarray(baseline['predictions']['G_signals']) != 'neutral').astype(int)
        print(pd.DataFrame([metrics(result['y'], b), result['metrics']], index=['G_signals', 'Binary feedback']).round(4))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, norm in zip(axes, [None, 'true']):
        ConfusionMatrixDisplay.from_predictions(
            result['y'],
            result['predictions'],
            labels=[0, 1],
            display_labels=['neutral', 'non-neutral'],
            normalize=norm,
            values_format='d' if norm is None else '.2f',
            ax=ax,
            colorbar=False,
        )
    fig.tight_layout()
    plt.show()
