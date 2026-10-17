from pathlib import Path
import hashlib
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
CLASSES = ['negative', 'neutral', 'positive']
RESULT_DIR = Path('artifacts/course_ovr')
CONFIGS = [(c, weight) for c in (0.1, 0.3, 1.0) for weight in (None, 'balanced')]
THRESHOLDS = [-0.5, 0.0, 0.5, 1.0]


def interactions(dialogs, training=False):
    records = []
    for d, dialog in enumerate(dialogs.to_dict('records')):
        turns = dialog['turns']
        users = [i for i, t in enumerate(turns) if t['role'] == 'user']
        for j, i in enumerate(users[1:], 1):
            previous = turns[i - 1] if i else {}
            following = turns[i + 1] if i + 1 < len(turns) else {}
            user = turns[i]['text']
            row = dict(
                dialog_pos=d,
                user=user,
                assistant=previous.get('text', '') if previous.get('role') == 'assistant' else '',
                following=following.get('text', '')[:600] if following.get('role') == 'assistant' else '',
                position=j / max(len(users) - 1, 1),
                log_length=np.log1p(len(user)),
                n_users=len(users),
                questions=user.count('?'),
                exclamations=user.count('!'),
            )
            if training:
                row['reaction'] = dialog['labels']['user_turns'][j]['reaction']
                if row['reaction'] not in CLASSES:
                    raise ValueError('Missing or invalid training reaction')
            records.append(row)
    return pd.DataFrame(records, columns=['dialog_pos', 'user', 'assistant', 'following', 'position', 'log_length', 'n_users', 'questions', 'exclamations'] + (['reaction'] if training else []))


def encoder(next_assistant=True):

    def word(limit):
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=limit, sublinear_tf=True, dtype=np.float32)
    blocks = [
        ('user_word', word(12000), 'user'),
        ('user_char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=18000, sublinear_tf=True, dtype=np.float32), 'user'),
        ('assistant', word(6000), 'assistant'),
        ('structure', StandardScaler(with_mean=False), ['position', 'log_length', 'n_users', 'questions', 'exclamations']),
    ]
    weights = {'user_word': 1.0, 'user_char': 1.0, 'assistant': 0.4, 'structure': 0.1}
    if next_assistant:
        blocks.append(('following', word(6000), 'following'))
        weights['following'] = 0.4
    return ColumnTransformer(blocks, transformer_weights=weights, sparse_threshold=1.0)


def fit_binary(x, reactions, label, config):
    c, weight = config
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = LinearSVC(C=c, class_weight=weight, dual=False, max_iter=5000, random_state=42)
        model.fit(x, reactions == label)
    scale = max(float(np.std(model.decision_function(x))), 1e-06)
    return (model, scale)


def aggregate(pairs, positive, negative, n_dialogs, thresholds):
    tp, tn = thresholds
    ep, en = (positive - tp, negative - tn)
    found = (ep >= 0) | (en >= 0)
    labels = np.where(ep > en, 'positive', 'negative')
    result = np.full(n_dialogs, 'neutral', dtype=object)
    if found.any():
        indices = np.flatnonzero(found)
        ids = pairs.dialog_pos.to_numpy()[indices].astype(int)
        last = np.full(n_dialogs, -1)
        np.maximum.at(last, ids, indices)
        present = last >= 0
        result[present] = labels[last[present]]
    return result


def select(training, configs, next_assistant, inner_splits):
    pairs = interactions(training, training=True)
    scores = {label: np.full((len(configs), len(pairs)), np.nan) for label in ('positive', 'negative')}
    cv = StratifiedGroupKFold(inner_splits, shuffle=True, random_state=42)
    for tr, va in cv.split(training, training.reaction, training.group):
        assert not set(training.group.iloc[tr]) & set(training.group.iloc[va])
        train_pairs = interactions(training.iloc[tr], training=True)
        val_pairs = interactions(training.iloc[va].drop(columns=['labels', 'reaction']), training=False)
        enc = encoder(next_assistant)
        xt = enc.fit_transform(train_pairs)
        xv = enc.transform(val_pairs)
        mask = pairs.dialog_pos.isin(va).to_numpy()
        assert mask.sum() == len(val_pairs)
        for label in scores:
            for k, config in enumerate(configs):
                model, scale = fit_binary(xt, train_pairs.reaction.to_numpy(), label, config)
                scores[label][k, mask] = model.decision_function(xv) / scale
    assert all((np.isfinite(s).all() for s in scores.values()))
    best = None
    rows = []
    for kp, cp in enumerate(configs):
        for kn, cn in enumerate(configs):
            for tp in THRESHOLDS:
                for tn in THRESHOLDS:
                    pred = aggregate(
                        pairs,
                        scores['positive'][kp],
                        scores['negative'][kn],
                        len(training),
                        (tp, tn),
                    )
                    value = f1_score(
                        training.reaction,
                        pred,
                        labels=CLASSES,
                        average='macro',
                        zero_division=0,
                    )
                    rows.append(dict(positive_C=cp[0], positive_weight=cp[1], negative_C=cn[0], negative_weight=cn[1], positive_threshold=tp, negative_threshold=tn, macro_f1=value))
                    if best is None or value > best[0]:
                        best = (value, cp, cn, (tp, tn))
    return (best, pd.DataFrame(rows))


def fit_final(training, choice, next_assistant):
    pairs = interactions(training, training=True)
    enc = encoder(next_assistant)
    x = enc.fit_transform(pairs)
    models = {label: fit_binary(x, pairs.reaction.to_numpy(), label, config) for label, config in [('positive', choice[1]), ('negative', choice[2])]}
    return dict(encoder=enc, models=models, thresholds=choice[3], next_assistant=next_assistant, train_dialog_ids=training.dialog_id.tolist())


def predict(bundle, dialogs):
    dialogs = dialogs[['dialog_id', 'turns']].reset_index(drop=True)
    pairs = interactions(dialogs)
    if pairs.empty:
        return np.full(len(dialogs), 'neutral', dtype=object)
    x = bundle['encoder'].transform(pairs)
    scores = {label: model.decision_function(x) / scale for label, (model, scale) in bundle['models'].items()}
    return aggregate(pairs, scores['positive'], scores['negative'], len(dialogs), bundle['thresholds'])


def fingerprint(training, next_assistant, configs, outer_splits, inner_splits):
    payload = training[['dialog_id', 'turns', 'labels', 'reaction', 'group']].to_json(orient='records')
    settings = repr((next_assistant, configs, outer_splits, inner_splits))
    return hashlib.sha256(Path(__file__).read_bytes() + payload.encode() + settings.encode()).hexdigest()


def run(training, next_assistant=True, configs=CONFIGS, outer_splits=5, inner_splits=3):
    training = training.reset_index(drop=True)
    destination = RESULT_DIR / ('next_assistant' if next_assistant else 'local')
    destination.mkdir(parents=True, exist_ok=True)
    key = fingerprint(training, next_assistant, configs, outer_splits, inner_splits)
    path = destination / 'results.joblib'
    if path.exists():
        cached = joblib.load(path)
        if cached.get('fingerprint') == key:
            return cached
    pred = np.full(len(training), '', dtype=object)
    fold_scores = []
    with threadpool_limits(limits=1):
        cv = StratifiedGroupKFold(outer_splits, shuffle=True, random_state=42)
        for fold, (tr, va) in enumerate(cv.split(training, training.reaction, training.group), 1):
            assert not set(training.group.iloc[tr]) & set(training.group.iloc[va])
            fit = training.iloc[tr].reset_index(drop=True)
            choice, search = select(fit, configs, next_assistant, inner_splits)
            search.to_csv(destination / f'threshold_search_fold_{fold}.csv', index=False)
            bundle = fit_final(fit, choice, next_assistant)
            pred[va] = predict(bundle, training.iloc[va])
            value = f1_score(
                training.reaction.iloc[va],
                pred[va],
                labels=CLASSES,
                average='macro',
                zero_division=0,
            )
            fold_scores.append(value)
        choice, search = select(training, configs, next_assistant, inner_splits)
        search.to_csv(destination / 'final_threshold_search.csv', index=False)
        bundle = fit_final(training, choice, next_assistant)
    joblib.dump(bundle, destination / 'final_model.joblib')
    result = dict(
        fingerprint=key,
        predictions=pred,
        dialog_ids=training.dialog_id.tolist(),
        y=training.reaction.to_numpy(),
        fold_scores=fold_scores,
        macro_f1=f1_score(training.reaction, pred, labels=CLASSES, average='macro', zero_division=0),
        mean_cv_macro_f1=float(np.mean(fold_scores)),
        final_choice=choice,
    )
    joblib.dump(result, path)
    return result


def display_result(result):
    import matplotlib.pyplot as plt
    print(f"OOF Macro-F1: {result['macro_f1']:.4f}; mean CV: {result['mean_cv_macro_f1']:.4f}")
    print(classification_report(result['y'], result['predictions'], labels=CLASSES, zero_division=0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, normalize in zip(axes, [None, 'true']):
        ConfusionMatrixDisplay.from_predictions(
            result['y'],
            result['predictions'],
            labels=CLASSES,
            normalize=normalize,
            values_format='d' if normalize is None else '.2f',
            ax=ax,
            colorbar=False,
        )
        ax.set_title('Число диалогов' if normalize is None else 'Доля истинного класса')
    plt.tight_layout()
    plt.show()
