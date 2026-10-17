from pathlib import Path
import json
import time
import hashlib
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
CLASSES = np.array(['negative', 'neutral', 'positive'])
hierarchy_classes = CLASSES
STATES = np.array(['continuation', 'feedback', 'newtopic', 'refinement', 'other'])
REASONS = [
    'Factual_Error',
    'Revision',
    'Gratitude',
    'Insufficient_Detail',
    'Negative_Feedback',
    'Ignored',
    'Acknowledgment',
    'Praise',
]
RESULT_DIR = Path('artifacts/reaction_experiments')


def metrics(y, pred):
    return {'Accuracy': float(accuracy_score(y, pred)), 'Balanced Accuracy': float(balanced_accuracy_score(y, pred)), 'Macro-F1': float(f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0))}


def last_non_neutral(reactions):
    return next((r for r in reversed(reactions) if r != 'neutral'), 'neutral')


def oracle_sequences(dialogs):
    return [[t['reaction'] for t in row.labels['user_turns'] if t.get('reaction') is not None] for row in dialogs.itertuples(index=False)]


def oracle_features(sequences):
    rows = []
    for seq in sequences:
        row = {'n': len(seq)}
        for c in CLASSES:
            positions = [i / max(len(seq) - 1, 1) for i, label in enumerate(seq) if label == c]
            row.update({c + '_count': seq.count(c), c + '_share': seq.count(c) / len(seq), c + '_last': int(seq[-1] == c), c + '_first': int(seq[0] == c), c + '_last_position': positions[-1] if positions else -1})
        row['change_share'] = np.mean(np.array(seq[1:]) != np.array(seq[:-1])) if len(seq) > 1 else 0
        for position in range(9):
            for c in CLASSES:
                row[f'position_{position}_{c}'] = int(position < len(seq) and seq[position] == c)
        rows.append(row)
    return pd.DataFrame(rows)


def run_a(dialogs, groups, n_splits=5):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dialogs = dialogs.reset_index(drop=True)
    y = dialogs.reaction.to_numpy()
    seqs = oracle_sequences(dialogs)
    mixed = np.array(['negative' in seq and 'positive' in seq for seq in seqs])
    rules = {
        'Last non-neutral': [last_non_neutral(s) for s in seqs],
        'First non-neutral': [next((r for r in s if r != 'neutral'), 'neutral') for s in seqs],
        'Negative priority': ['negative' if 'negative' in s else 'positive' if 'positive' in s else 'neutral' for s in seqs],
        'Positive priority': ['positive' if 'positive' in s else 'negative' if 'negative' in s else 'neutral' for s in seqs],
        'Last reaction': [s[-1] for s in seqs],
    }
    features = oracle_features(seqs)
    cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)
    learned = {
        'Oracle tree depth 3': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Oracle tree depth 5': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Oracle LR': Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(C=1, max_iter=2000, class_weight='balanced'))]),
    }
    predictions = {name: np.asarray(pred, dtype=object) for name, pred in rules.items()}
    for name, model in learned.items():
        pred = np.full(len(y), '', dtype=object)
        for tr, va in cv.split(features, y, groups):
            assert not set(np.asarray(groups)[tr]) & set(np.asarray(groups)[va])
            fitted = clone(model).fit(features.iloc[tr], y[tr])
            pred[va] = fitted.predict(features.iloc[va])
        predictions[name] = pred
    table = pd.DataFrame([{'Method': name, **metrics(y, pred), 'Errors': int(np.sum(y != pred)), 'Mixed accuracy': float(np.mean(y[mixed] == pred[mixed])) if mixed.any() else np.nan} for name, pred in predictions.items()]).set_index('Method')
    mixed_cases = pd.DataFrame({'dialog_id': dialogs.dialog_id, 'sequence': [' → '.join(s) for s in seqs], 'target': y, 'rule': predictions['Last non-neutral']})[mixed]
    table.to_csv(RESULT_DIR / 'A_metrics.csv')
    mixed_cases.to_csv(RESULT_DIR / 'A_mixed.csv', index=False)
    print('Правило по истинным реакциям реплик')
    print(table.round(4).to_string())
    print('Диалогов с обоими знаками:', int(mixed.sum()))
    return (table, mixed_cases)


def make_pairs(dialogs, labels=True):
    rows = []
    for d, row in enumerate(dialogs.itertuples(index=False)):
        users = [i for i, t in enumerate(row.turns) if t['role'] == 'user']
        for user_index, turn_idx in enumerate(users[1:], start=1):
            if row.turns[turn_idx - 1]['role'] != 'assistant':
                raise ValueError('Invalid pair')
            current = row.turns[turn_idx]['text']
            assistant = row.turns[turn_idx - 1]['text']
            previous = row.turns[users[user_index - 1]]['text']
            record = {
                'dialog_pos': d,
                'user': current,
                'assistant': assistant,
                'previous_user': previous,
                'position': user_index / max(len(users) - 1, 1),
                'log_user_chars': np.log1p(len(current)),
                'log_assistant_chars': np.log1p(len(assistant)),
                'log_previous_chars': np.log1p(len(previous)),
                'log_user_words': np.log1p(len(current.split())),
                'question_marks': current.count('?'),
                'exclamations': current.count('!'),
                'uppercase_share': sum((c.isupper() for c in current)) / max(len(current), 1),
                'digit_share': sum((c.isdigit() for c in current)) / max(len(current), 1),
                'n_user_turns': len(users),
            }
            if labels:
                annotation = row.labels['user_turns'][user_index]
                record['reaction'] = annotation['reaction']
                record['state'] = annotation['state'] if annotation['state'] in STATES else 'other'
                record['reason_set'] = {v.strip() for v in (annotation.get('reaction_reason') or '').split(';') if v.strip()}
            rows.append(record)
    return pd.DataFrame(rows)
NUMERIC = [
    'position',
    'log_user_chars',
    'log_assistant_chars',
    'log_previous_chars',
    'log_user_words',
    'question_marks',
    'exclamations',
    'uppercase_share',
    'digit_share',
    'n_user_turns',
]
WEIGHTS = {'equal': (1.0, 1.0, 1.0, 1.0), 'user_focus': (1.0, 0.5, 0.75, 0.25)}


def encoder():

    def word():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000, sublinear_tf=True, dtype=np.float32)
    return ColumnTransformer([('user_word', word(), 'user'), ('user_char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=10000, sublinear_tf=True, dtype=np.float32), 'user'), ('assistant_word', word(), 'assistant'), ('previous_word', word(), 'previous_user'), ('numeric', StandardScaler(), NUMERIC)], sparse_threshold=1.0)


def weighted_features(x, fitted_encoder, weights):
    user, assistant, previous, numeric = weights
    scales = np.ones(x.shape[1], dtype=np.float32)
    for name, w in [('user_word', user), ('user_char', user), ('assistant_word', assistant), ('previous_word', previous), ('numeric', numeric)]:
        scales[fitted_encoder.output_indices_[name]] = w
    return sparse.csr_matrix(x.multiply(scales), dtype=np.float32)


def aligned_proba(model, x, classes):
    result = np.zeros((x.shape[0], len(classes)))
    raw = model.predict_proba(x)
    for i, c in enumerate(model.classes_):
        if c in classes:
            result[:, list(classes).index(c)] = raw[:, i]
    return result


class CalibratedSVC:
    def fit(self, x, y, cal_x, cal_y):
        y = np.asarray(y)
        cal_y = np.asarray(cal_y)
        if len(np.unique(y)) == 1:
            self.base = DummyClassifier(strategy='prior').fit(x, y)
            self.cal = None
            self.classes_ = self.base.classes_
            return self
        self.base = LinearSVC(C=0.3, class_weight='balanced', max_iter=5000, random_state=42).fit(x, y)
        if len(np.unique(cal_y)) < 2:
            self.cal = DummyClassifier(strategy='prior').fit(self._margins(cal_x), cal_y)
        else:
            self.cal = LogisticRegression(
                C=1,
                max_iter=1500,
                random_state=42,
            ).fit(self._margins(cal_x), cal_y)
        self.classes_ = self.cal.classes_
        return self

    def _margins(self, x):
        margins = self.base.decision_function(x)
        return margins[:, None] if margins.ndim == 1 else margins

    def predict_proba(self, x):
        return self.base.predict_proba(x) if self.cal is None else self.cal.predict_proba(self._margins(x))


def calibrated_predictions(core_x, cal_x, target_x, core, cal, aux=True):
    direct = CalibratedSVC().fit(core_x, core.reaction, cal_x, cal.reaction)
    probabilities = aligned_proba(direct, target_x, CLASSES)
    core_stage1 = np.where(core.reaction == 'neutral', 'neutral', 'non-neutral')
    cal_stage1 = np.where(cal.reaction == 'neutral', 'neutral', 'non-neutral')
    stage1 = CalibratedSVC().fit(core_x, core_stage1, cal_x, cal_stage1)
    core_mask = (core.reaction != 'neutral').to_numpy()
    cal_mask = (cal.reaction != 'neutral').to_numpy()
    stage2 = CalibratedSVC().fit(core_x[core_mask], core.reaction[core_mask], cal_x[cal_mask], cal.reaction[cal_mask])
    gate = aligned_proba(stage1, target_x, np.array(['neutral', 'non-neutral']))
    sign = aligned_proba(stage2, target_x, np.array(['negative', 'positive']))
    two = np.column_stack([gate[:, 1] * sign[:, 0], gate[:, 0], gate[:, 1] * sign[:, 1]])
    auxiliary = None
    if aux:
        state = CalibratedSVC().fit(core_x, core.state, cal_x, cal.state)
        state_p = aligned_proba(state, target_x, STATES)
        reasons = []
        for reason in REASONS:
            core_y = np.array([int(reason in v) for v in core.reason_set])
            cal_y = np.array([int(reason in v) for v in cal.reason_set])
            head = CalibratedSVC().fit(core_x, core_y, cal_x, cal_y)
            reasons.append(aligned_proba(head, target_x, np.array([0, 1]))[:, 1])
        auxiliary = np.column_stack([state_p, *reasons])
    return (probabilities, two, auxiliary)


def predict_fold_pairs(fit_dialogs, target_dialogs, seed):
    split = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    core_idx, cal_idx = next(split.split(fit_dialogs, fit_dialogs.reaction, fit_dialogs.group))
    assert not set(fit_dialogs.group.iloc[core_idx]) & set(fit_dialogs.group.iloc[cal_idx])
    core = make_pairs(fit_dialogs.iloc[core_idx].reset_index(drop=True))
    cal = make_pairs(fit_dialogs.iloc[cal_idx].reset_index(drop=True))
    target = make_pairs(target_dialogs.reset_index(drop=True), labels=False)
    enc = encoder()
    xcore = enc.fit_transform(core)
    xcal = enc.transform(cal)
    xtarget = enc.transform(target)
    predictions = {}
    auxiliary = None
    for name, weights in WEIGHTS.items():
        wc = weighted_features(xcore, enc, weights)
        wa = weighted_features(xcal, enc, weights)
        wt = weighted_features(xtarget, enc, weights)
        direct, two, extra = calibrated_predictions(wc, wa, wt, core, cal, aux=name == 'user_focus')
        predictions[name + ' direct'] = direct
        predictions[name + ' two-stage'] = two
        if extra is not None:
            auxiliary = extra
    return (target, predictions, auxiliary)


def soft_last_non_neutral(pairs, p, n_dialogs):
    result = np.zeros((n_dialogs, 3))
    result[:, 1] = 1
    ids = pairs.dialog_pos.to_numpy()
    for i in range(len(p)):
        d = ids[i]
        result[d, 0] = p[i, 0] + p[i, 1] * result[d, 0]
        result[d, 2] = p[i, 2] + p[i, 1] * result[d, 2]
        result[d, 1] *= p[i, 1]
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-06)
    return result


def dialog_features(pairs, p, n_dialogs, extra=None):
    result = aggregate_reaction_proba(pairs, p, n_dialogs)
    rule = soft_last_non_neutral(pairs, p, n_dialogs)
    for k, c in enumerate(CLASSES):
        result['last_rule_' + c] = rule[:, k]
    if extra is not None:
        ids = pairs.dialog_pos.to_numpy()
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        counts = np.diff(np.r_[starts, len(ids)])
        ends = starts + counts - 1
        names = list(STATES) + REASONS
        means = np.add.reduceat(extra, starts, axis=0) / counts[:, None]
        maxima = np.maximum.reduceat(extra, starts, axis=0)
        for k, name in enumerate(names):
            result['aux_' + name + '_mean'] = means[:, k]
            result['aux_' + name + '_max'] = maxima[:, k]
            result['aux_' + name + '_last'] = extra[ends, k]
    return result


def meta_model():
    return Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(C=1, class_weight='balanced', max_iter=1500, random_state=42))])


def presence_labels(dialogs):
    sequences = oracle_sequences(dialogs)
    return np.array([['negative' in s, 'positive' in s] for s in sequences])


def fit_binary(features, y):
    return DummyClassifier(strategy='prior').fit(features, y) if len(np.unique(y)) < 2 else meta_model().fit(features, y)


def binary_p(model, x):
    return aligned_proba(model, x, np.array([False, True]))[:, 1]


def structured_predict(fit_features, val_features, fit_dialogs):
    presence = presence_labels(fit_dialogs)
    any_nn = presence.any(axis=1)
    both = presence.all(axis=1)
    gate = fit_binary(fit_features, any_nn)
    negative = fit_binary(fit_features, presence[:, 0])
    positive = fit_binary(fit_features, presence[:, 1])
    conflict = fit_binary(
        fit_features.loc[both],
        fit_dialogs.reaction.to_numpy()[both] == 'positive',
    ) if both.any() else None
    routed = binary_p(gate, val_features) >= 0.5
    neg = binary_p(negative, val_features) >= 0.5
    pos = binary_p(positive, val_features) >= 0.5
    rule = val_features[['last_rule_negative', 'last_rule_neutral', 'last_rule_positive']].to_numpy()
    pred = np.full(len(val_features), 'neutral', dtype=object)
    pred[routed & neg & ~pos] = 'negative'
    pred[routed & pos & ~neg] = 'positive'
    mixed = routed & neg & pos
    if mixed.any():
        conflict_positive = binary_p(
            conflict,
            val_features.loc[mixed],
        ) >= 0.5 if conflict is not None else rule[mixed, 2] > rule[mixed, 0]
        pred[mixed] = np.where(conflict_positive, 'positive', 'negative')
    uncertain = routed & ~neg & ~pos
    if uncertain.any():
        pred[uncertain] = np.where(rule[uncertain, 2] > rule[uncertain, 0], 'positive', 'negative')
    return pred


def run_bcd(dialogs, groups, n_splits=5, inner_splits=3):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    dialogs = dialogs.reset_index(drop=True).copy()
    dialogs['group'] = np.asarray(groups)
    y = dialogs.reaction.to_numpy()
    all_pairs = make_pairs(dialogs)
    configs = [weight + ' ' + method for weight in WEIGHTS for method in ['direct', 'two-stage']]
    names = ['B ' + name for name in configs] + ['C auxiliary LR', 'C auxiliary HGB', 'D soft last rule', 'D structural']
    outer_predictions = {name: np.full(len(y), '', dtype=object) for name in names}
    pair_oof = {name: np.full((len(all_pairs), 3), np.nan) for name in configs}
    folds = {name: [] for name in names}
    timings = []
    outer = StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(outer.split(dialogs, y, groups), start=1):
        tick = time.perf_counter()
        assert not set(np.asarray(groups)[tr]) & set(np.asarray(groups)[va])
        fit = dialogs.iloc[tr].reset_index(drop=True)
        val = dialogs.iloc[va].reset_index(drop=True)
        fp = make_pairs(fit)
        inner_probs = {name: np.full((len(fp), 3), np.nan) for name in configs}
        inner_extra = np.full((len(fp), len(STATES) + len(REASONS)), np.nan)
        inner = StratifiedGroupKFold(inner_splits, shuffle=True, random_state=42)
        for inner_fold, (it, iv) in enumerate(inner.split(fit, fit.reaction, fit.group), start=1):
            target, prob, extra = predict_fold_pairs(
                fit.iloc[it].reset_index(drop=True),
                fit.iloc[iv].reset_index(drop=True),
                42 + inner_fold,
            )
            mask = fp.dialog_pos.isin(iv).to_numpy()
            assert np.all(np.diff(iv) > 0)
            for name in configs:
                inner_probs[name][mask] = prob[name]
            inner_extra[mask] = extra
        assert np.isfinite(inner_extra).all()
        vp, val_probs, val_extra = predict_fold_pairs(fit, val, 84)
        for name in configs:
            assert np.isfinite(inner_probs[name]).all()
            fit_feat = dialog_features(fp, inner_probs[name], len(fit))
            val_feat = dialog_features(vp, val_probs[name], len(val))
            pred = meta_model().fit(fit_feat, y[tr]).predict(val_feat)
            outer_predictions['B ' + name][va] = pred
            pair_mask = all_pairs.dialog_pos.isin(va).to_numpy()
            pair_oof[name][pair_mask] = val_probs[name]
        base = 'user_focus direct'
        fit_feat = dialog_features(fp, inner_probs[base], len(fit), inner_extra)
        val_feat = dialog_features(vp, val_probs[base], len(val), val_extra)
        outer_predictions['C auxiliary LR'][va] = meta_model().fit(fit_feat, y[tr]).predict(val_feat)
        hgb = HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=7,
            l2_regularization=1,
            class_weight='balanced',
            early_stopping=False,
            random_state=42,
        )
        outer_predictions['C auxiliary HGB'][va] = hgb.fit(fit_feat, y[tr]).predict(val_feat)
        soft = soft_last_non_neutral(vp, val_probs[base], len(val))
        outer_predictions['D soft last rule'][va] = CLASSES[soft.argmax(axis=1)]
        outer_predictions['D structural'][va] = structured_predict(fit_feat, val_feat, fit)
        for name in names:
            score = f1_score(y[va], outer_predictions[name][va], average='macro', zero_division=0)
            folds[name].append(float(score))
            print(f'  {name}: {score:.4f}')
        elapsed = time.perf_counter() - tick
        timings.append(elapsed)
        joblib.dump(
            {'predictions': outer_predictions, 'pair_probabilities': pair_oof, 'fold_scores': folds, 'completed_folds': fold, 'dialog_ids': dialogs.dialog_id.tolist()},
            RESULT_DIR / 'BCD_checkpoint.joblib',
        )
    table = pd.DataFrame([{'Method': name, **metrics(y, pred), 'Mean CV Macro-F1': np.mean(folds[name]), 'Std CV Macro-F1': np.std(folds[name])} for name, pred in outer_predictions.items()]).set_index('Method')
    pair_table = pd.DataFrame([{'Method': name, **metrics(all_pairs.reaction.to_numpy(), CLASSES[p.argmax(axis=1)])} for name, p in pair_oof.items()]).set_index('Method')
    table.to_csv(RESULT_DIR / 'BCD_dialog_metrics.csv')
    pair_table.to_csv(RESULT_DIR / 'B_pair_metrics.csv')
    reports = {name: classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0) for name, pred in outer_predictions.items()}
    mixed = presence_labels(dialogs).all(axis=1)
    mixed_table = pd.DataFrame([{'Method': name, 'Mixed accuracy': float(np.mean(y[mixed] == pred[mixed])), 'Mixed count': int(mixed.sum())} for name, pred in outer_predictions.items()]).set_index('Method')
    mixed_table.to_csv(RESULT_DIR / 'D_mixed_metrics.csv')
    summary = {
        'seconds': time.perf_counter() - start,
        'fold_seconds': timings,
        'reports': reports,
        'calibration': 'Held-out training groups; no true intermediate labels in validation features',
        'C': 0.3,
        'weights': WEIGHTS,
        'inner_splits': inner_splits,
        'outer_splits': n_splits,
        'dialog_ids': dialogs.dialog_id.tolist(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (RESULT_DIR / 'BCD_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\nМетрики реплик:')
    print(pair_table.round(4).to_string())
    print('\nМетрики диалогов:')
    print(table.round(4).to_string())
    print('\nДиалоги с обоими знаками:')
    print(mixed_table.round(4).to_string())
    return (table, pair_table, mixed_table)


def aggregate_reaction_proba(pairs, probabilities, n_dialogs):
    ids = pairs['dialog_pos'].to_numpy(dtype=np.int64)
    positions = pairs['position'].to_numpy(dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(ids), len(hierarchy_classes)) or not np.isfinite(p).all():
        raise ValueError('Некорректные вероятности реакций')
    order = np.argsort(ids, kind='stable')
    ids, positions, p = (ids[order], positions[order], p[order])
    counts = np.bincount(ids, minlength=n_dialogs)
    if len(counts) != n_dialogs or (counts == 0).any():
        raise ValueError('Диалог не содержит реакций или имеет неверный индекс')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    ends = starts + counts - 1
    hard = p.argmax(axis=1)
    means = np.add.reduceat(p, starts, axis=0) / counts[:, None]
    maxima = np.maximum.reduceat(p, starts, axis=0)
    variances = np.add.reduceat((p - means[ids]) ** 2, starts, axis=0) / counts[:, None]
    previous = np.maximum(starts, ends - 1)
    last2 = (p[ends] + p[previous]) / 2
    last2[counts == 1] = p[ends[counts == 1]]
    row = {'n_reactions': counts}
    for k, name in enumerate(hierarchy_classes):
        selected = hard == k
        selected_counts = np.bincount(ids, weights=selected, minlength=n_dialogs)
        peak_candidates = np.where(p[:, k] == maxima[ids, k], np.arange(len(ids)), len(ids))
        peak_indices = np.minimum.reduceat(peak_candidates, starts)
        row.update({f'{name}_mean': means[:, k], f'{name}_max': maxima[:, k], f'{name}_std': np.sqrt(variances[:, k]), f'{name}_first': p[starts, k], f'{name}_last': p[ends, k], f'{name}_last2_mean': last2[:, k], f'{name}_last_minus_first': p[ends, k] - p[starts, k], f'{name}_count': selected_counts, f'{name}_share': selected_counts / counts, f'{name}_peak_position': positions[peak_indices]})
    non_neutral = hard != 1
    nn_counts = np.bincount(ids, weights=non_neutral, minlength=n_dialogs)
    row['non_neutral_count'] = nn_counts
    row['non_neutral_share'] = nn_counts / counts
    row['last_non_neutral_position'] = np.maximum.reduceat(np.where(non_neutral, positions, -1), starts)
    changes = np.r_[False, (ids[1:] == ids[:-1]) & (hard[1:] != hard[:-1])]
    row['reaction_change_share'] = np.bincount(ids, weights=changes, minlength=n_dialogs) / np.maximum(counts - 1, 1)
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1)), axis=1)
    row['mean_entropy'] = np.add.reduceat(entropy, starts) / counts
    return pd.DataFrame(row)
