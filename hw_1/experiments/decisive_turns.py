from pathlib import Path
import hashlib
import json
import re
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from .reaction_experiments import make_pairs, metrics, CLASSES, presence_labels
from .fca_reactions import BinaryInteractionEncoder, PATTERNS, data_fingerprint, source_fingerprint as dependency_hash
RESULT_DIR = Path('artifacts/decisive_turns')
THRESHOLDS = np.array([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
VARIANTS = [
    'Local / decisive sign',
    'Full / decisive sign',
    'Local / all-nonneutral sign',
    'Full / all-nonneutral sign',
]


def source_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes() + dependency_hash().encode()).hexdigest()


def contexts(dialogs, labels=False):
    pairs = make_pairs(dialogs, labels=labels)
    records = []
    for row in dialogs.itertuples(index=False):
        users = [i for i, t in enumerate(row.turns) if t['role'] == 'user']
        for user_index, turn_index in enumerate(users[1:], start=1):
            following = row.turns[turn_index + 1] if turn_index + 1 < len(row.turns) else None
            records.append({'following_assistant': following['text'] if following is not None and following['role'] == 'assistant' else '', 'future_users': '\n'.join((row.turns[i]['text'] for i in users[user_index + 1:])), 'later_assistants': '\n'.join((t['text'] for t in row.turns[turn_index + 2:] if t['role'] == 'assistant')), 'future_user_count': len(users) - user_index - 1})
    extra = pd.DataFrame(records)
    for column in extra:
        pairs[column] = extra[column].to_numpy()
    return pairs


def gold_choices(dialogs):
    choices = []
    for row in dialogs.itertuples(index=False):
        reactions = [t.get('reaction') for t in row.labels['user_turns'][1:]]
        eligible = [i for i, r in enumerate(reactions) if r in ['negative', 'positive']]
        choices.append(eligible[-1] if eligible else -1)
    return np.array(choices)


def decisive_targets(dialogs, pairs):
    gold = gold_choices(dialogs)
    counts = np.bincount(pairs.dialog_pos.to_numpy(), minlength=len(dialogs))
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    target = np.zeros(len(pairs), dtype=int)
    nonneutral = gold >= 0
    target[starts[nonneutral] + gold[nonneutral]] = 1
    assert target.sum() == np.sum(dialogs.reaction.to_numpy() != 'neutral')
    assert np.all(dialogs.reaction.to_numpy()[nonneutral] == pairs.reaction.to_numpy()[starts[nonneutral] + gold[nonneutral]])
    return target


class ContextEncoder:
    def _future_flags(self, pairs):
        output = []
        for source in ['following_assistant', 'future_users', 'later_assistants']:
            texts = pairs[source].fillna('').str.lower().to_numpy()
            for pattern in PATTERNS.values():
                regex = re.compile(pattern)
                output.append(np.array([bool(regex.search(t)) for t in texts]))
        count = pairs.future_user_count.to_numpy()
        output.extend([count == 0, count == 1, count == 2, count >= 3])
        return sparse.csr_matrix(np.column_stack(output).astype(np.float32))

    def fit_transform(self, pairs):
        self.local = BinaryInteractionEncoder()
        local = self.local.fit_transform(pairs)

        def word():
            return CountVectorizer(binary=True, ngram_range=(1, 2), min_df=3, max_features=3000, dtype=np.float32)
        self.future = ColumnTransformer(
            [('following', word(), 'following_assistant'), ('users', word(), 'future_users'), ('assistants', word(), 'later_assistants')],
            sparse_threshold=1.0,
        )
        future = self.future.fit_transform(pairs)
        full = sparse.hstack([local, future, self._future_flags(pairs)], format='csr', dtype=np.float32)
        return (Normalizer(copy=False).fit_transform(local), Normalizer(copy=False).fit_transform(full))

    def transform(self, pairs):
        local = self.local.transform(pairs)
        full = sparse.hstack(
            [local, self.future.transform(pairs), self._future_flags(pairs)],
            format='csr',
            dtype=np.float32,
        )
        return (Normalizer(copy=False).transform(local), Normalizer(copy=False).transform(full))


def fit_svc(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = LinearSVC(
            C=0.3,
            class_weight='balanced',
            dual=False,
            max_iter=5000,
            random_state=42,
        ).fit(x, y)
    return model


class DecisiveModels:
    def fit(self, dialogs):
        pairs = contexts(dialogs, labels=True)
        target = decisive_targets(dialogs, pairs)
        self.encoder = ContextEncoder()
        local, full = self.encoder.fit_transform(pairs)
        self.selectors = {'Local': fit_svc(local, target), 'Full': fit_svc(full, target)}
        masks = {
            'decisive sign': target.astype(bool),
            'all-nonneutral sign': pairs.reaction.isin(['negative', 'positive']).to_numpy(),
        }
        self.signs = {name: fit_svc(local[mask], pairs.reaction.to_numpy()[mask]) for name, mask in masks.items()}
        self.n_iterations_ = {
            **{name + ' selector': m.n_iter_ for name, m in self.selectors.items()},
            **{name: m.n_iter_ for name, m in self.signs.items()},
        }
        return self

    def scores_and_signs(self, dialogs):
        pairs = contexts(dialogs, labels=False)
        local, full = self.encoder.transform(pairs)
        scores = {
            'Local': self.selectors['Local'].decision_function(local),
            'Full': self.selectors['Full'].decision_function(full),
        }
        signs = {name: model.predict(local) for name, model in self.signs.items()}
        return (pairs, scores, signs)


def select_and_classify(pairs, scores, signs, n_dialogs, threshold):
    ids = pairs.dialog_pos.to_numpy()
    counts = np.bincount(ids, minlength=n_dialogs)
    if (counts == 0).any():
        raise ValueError('A dialogue has no candidate user turns')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    best = np.maximum.reduceat(scores, starts)
    indices = np.maximum.reduceat(np.where(scores == best[ids], np.arange(len(ids)), -1), starts)
    found = best >= threshold
    pred = np.full(n_dialogs, 'neutral', dtype=object)
    pred[found] = signs[indices[found]]
    chosen = np.full(n_dialogs, -1, dtype=int)
    chosen[found] = indices[found] - starts[found]
    return (pred, chosen)


def crossfit_training(fit_dialogs, inner_splits=3):
    pairs = contexts(fit_dialogs, labels=False)
    scores = {name: np.full(len(pairs), np.nan) for name in ['Local', 'Full']}
    signs = {name: np.full(len(pairs), '', dtype=object) for name in ['decisive sign', 'all-nonneutral sign']}
    inner = StratifiedGroupKFold(inner_splits, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(inner.split(fit_dialogs, fit_dialogs.reaction, fit_dialogs.group), start=1):
        assert not set(fit_dialogs.group.iloc[tr]) & set(fit_dialogs.group.iloc[va])
        fitted = DecisiveModels().fit(fit_dialogs.iloc[tr].reset_index(drop=True))
        val = fit_dialogs.iloc[va].drop(columns=['labels', 'reaction']).reset_index(drop=True)
        _, sv, sg = fitted.scores_and_signs(val)
        mask = pairs.dialog_pos.isin(va).to_numpy()
        assert np.all(np.diff(va) > 0)
        for name in scores:
            scores[name][mask] = sv[name]
        for name in signs:
            signs[name][mask] = sg[name]
    assert all((np.isfinite(v).all() for v in scores.values()))
    assert all(((v != '').all() for v in signs.values()))
    thresholds = {}
    search = []
    for variant in VARIANTS:
        selector, sign = variant.split(' / ')
        best_score = -1
        best_threshold = None
        for threshold in THRESHOLDS:
            pred, _ = select_and_classify(pairs, scores[selector], signs[sign], len(fit_dialogs), threshold)
            score = f1_score(fit_dialogs.reaction, pred, labels=CLASSES, average='macro', zero_division=0)
            search.append({'variant': variant, 'threshold': float(threshold), 'inner_macro_f1': float(score)})
            if score > best_score + 1e-12:
                best_score = score
                best_threshold = float(threshold)
        thresholds[variant] = best_threshold
    return (thresholds, pd.DataFrame(search))


def predict_decisive(bundle, dialogs, variant='Full / decisive sign', return_positions=False):
    if variant not in VARIANTS:
        raise ValueError('Unknown decisive variant')
    if len(dialogs) == 0:
        result = np.empty(0, dtype=object)
        positions = np.empty(0, dtype=int)
        return (result, positions) if return_positions else result
    pairs, scores, signs = bundle['models'].scores_and_signs(dialogs.reset_index(drop=True))
    selector, sign = variant.split(' / ')
    pred, chosen = select_and_classify(
        pairs,
        scores[selector],
        signs[sign],
        len(dialogs),
        bundle['thresholds'][variant],
    )
    public = np.where(chosen >= 0, chosen + 1, -1)
    return (pred, public) if return_positions else pred

@threadpool_limits.wrap(limits=1)


def run_experiment(dialogs, groups, n_splits=5, inner_splits=3, train_final=True):
    start = time.perf_counter()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dialogs = dialogs.reset_index(drop=True).copy()
    dialogs['group'] = np.asarray(groups)
    y = dialogs.reaction.to_numpy()
    gold = gold_choices(dialogs)
    predictions = {name: np.full(len(dialogs), '', dtype=object) for name in VARIANTS}
    choices = {name: np.full(len(dialogs), -2, dtype=int) for name in VARIANTS}
    folds = {name: [] for name in VARIANTS}
    threshold_tables = []
    outer = StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(outer.split(dialogs, y, groups), start=1):
        tick = time.perf_counter()
        assert not set(np.asarray(groups)[tr]) & set(np.asarray(groups)[va])
        fit = dialogs.iloc[tr].reset_index(drop=True)
        thresholds, search = crossfit_training(fit, inner_splits)
        search['outer_fold'] = fold
        threshold_tables.append(search)
        model = DecisiveModels().fit(fit)
        val = dialogs.iloc[va].drop(columns=['labels', 'reaction']).reset_index(drop=True)
        pairs, scores, signs = model.scores_and_signs(val)
        for name in VARIANTS:
            selector, sign = name.split(' / ')
            pred, chosen = select_and_classify(
                pairs,
                scores[selector],
                signs[sign],
                len(va),
                thresholds[name],
            )
            predictions[name][va] = pred
            choices[name][va] = chosen
            score = f1_score(y[va], pred, labels=CLASSES, average='macro', zero_division=0)
            folds[name].append(float(score))
            print(f'  {name}: threshold={thresholds[name]:.1f}, Macro-F1={score:.4f}')
        joblib.dump(
            {'completed_folds': fold, 'predictions': predictions, 'choices': choices, 'fold_scores': folds, 'dialog_ids': dialogs.dialog_id.tolist()},
            RESULT_DIR / 'checkpoint.joblib',
        )
    table = pd.DataFrame([{'Method': name, **metrics(y, pred), 'Mean CV Macro-F1': np.mean(folds[name]), 'Std CV Macro-F1': np.std(folds[name])} for name, pred in predictions.items()]).set_index('Method')
    diagnostics = []
    for name, selected in choices.items():
        found = selected >= 0
        nonneutral = gold >= 0
        exact = (selected == gold) & nonneutral
        diagnostics.append({'Method': name, 'Exact decisive recall': float(np.mean(selected[nonneutral] == gold[nonneutral])), 'Chosen decisive precision': float(exact.sum() / max(found.sum(), 1)), 'Neutral correctly absent': float(np.mean(selected[~nonneutral] == -1)), 'Nonneutral found': float(found[nonneutral].mean())})
    diagnostic = pd.DataFrame(diagnostics).set_index('Method')
    mixed = presence_labels(dialogs).all(axis=1)
    mixed_table = pd.DataFrame([{'Method': name, 'Mixed accuracy': float(np.mean(y[mixed] == pred[mixed]))} for name, pred in predictions.items()]).set_index('Method')
    table.to_csv(RESULT_DIR / 'dialog_metrics.csv')
    diagnostic.to_csv(RESULT_DIR / 'selector_metrics.csv')
    mixed_table.to_csv(RESULT_DIR / 'mixed_metrics.csv')
    pd.concat(threshold_tables).to_csv(RESULT_DIR / 'threshold_search.csv', index=False)
    reports = {name: classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0) for name, pred in predictions.items()}
    bundle = None
    if train_final:
        thresholds, final_search = crossfit_training(dialogs, inner_splits)
        bundle = {
            'models': DecisiveModels().fit(dialogs),
            'thresholds': thresholds,
            'train_dialog_ids': dialogs.dialog_id.tolist(),
        }
        joblib.dump(bundle, RESULT_DIR / 'final_model.joblib')
        final_search.to_csv(RESULT_DIR / 'final_threshold_search.csv', index=False)
    summary = {
        'seconds': time.perf_counter() - start,
        'reports': reports,
        'fold_scores': folds,
        'dialog_ids': dialogs.dialog_id.tolist(),
        'source_sha256': source_fingerprint(),
        'data_sha256': data_fingerprint(dialogs, groups),
        'outer_splits': n_splits,
        'inner_splits': inner_splits,
        'final_trained': train_final,
        'thresholds': bundle['thresholds'] if bundle is not None else None,
    }
    (RESULT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\nМетрики диалогов:\n' + table.round(4).to_string())
    print('\nПоиск решающей реплики:\n' + diagnostic.round(4).to_string())
    print('\nДиалоги с обоими знаками:\n' + mixed_table.round(4).to_string())
    return (table, diagnostic, mixed_table, bundle)
