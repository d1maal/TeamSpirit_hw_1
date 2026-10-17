from pathlib import Path
from itertools import combinations
import hashlib
import json
import re
import warnings
import time
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from .reaction_experiments import CLASSES, make_pairs, metrics, soft_last_non_neutral, last_non_neutral, presence_labels
RESULT_DIR = Path('artifacts/fca_reactions')
PATTERNS = {
    'gratitude': r'\b(thanks?|thank\s+you|appreciate|grateful)\b|спасибо|благодар',
    'praise': r'\b(great|excellent|perfect|awesome|helpful|wonderful|brilliant)\b|отлично|прекрасно',
    'error': r'\b(wrong|incorrect|mistake|error|broken|false|inaccurate)\b|ошибк|неверн|неправиль',
    'failure': r"doesn[’']?t\s+work|not\s+work|still\s+(wrong|not|broken)|не\s+работает",
    'revision': r'\b(rewrite|redo|revise|correct|fix|change|modify|again)\b|исправ|передел|перепиш',
    'detail': r'\b(explain|details?|specific|elaborate|clarify|examples?)\b|подроб|поясни|пример',
    'negation': r"\b(no|not|never|nothing|cannot|can[’']?t|don[’']?t|doesn[’']?t)\b|\b(нет|не)\b",
    'agreement': r'\b(ok|okay|yes|sure|understood|got\s+it)\b|понятно|хорошо',
    'apology': r'\b(sorry|apologize|apologies|apology)\b|извини|извиня',
}


class BinaryInteractionEncoder:
    def _manual(self, pairs):
        columns = []
        names = []
        for source in ['user', 'assistant', 'previous_user']:
            texts = pairs[source].fillna('').str.lower().to_numpy()
            for name, pattern in PATTERNS.items():
                regex = re.compile(pattern)
                columns.append(np.array([bool(regex.search(t)) for t in texts]))
                names.append(source + '_' + name)
        user = pairs.user.fillna('').to_numpy()
        length = np.array([len(t.split()) for t in user])
        features = {
            'user_question': np.array(['?' in t for t in user]),
            'user_exclamation': np.array(['!' in t for t in user]),
            'user_short': length <= 5,
            'user_long': length >= 80,
            'user_cyrillic': np.array([bool(re.search('[А-Яа-яЁё]', t)) for t in user]),
            'position_early': pairs.position.to_numpy() <= 1 / 3,
            'position_middle': (pairs.position.to_numpy() > 1 / 3) & (pairs.position.to_numpy() <= 2 / 3),
            'position_late': pairs.position.to_numpy() > 2 / 3,
            'position_last': pairs.position.to_numpy() == 1,
        }
        columns.extend(features.values())
        names.extend(features)
        return (np.column_stack(columns).astype(np.float32), names)

    def fit_transform(self, pairs):

        def word(cap):
            return CountVectorizer(binary=True, ngram_range=(1, 2), min_df=3, max_features=cap, dtype=np.float32)
        self.text = ColumnTransformer(
            [('user_word', word(5000), 'user'), ('user_char', CountVectorizer(binary=True, analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=5000, dtype=np.float32), 'user'), ('assistant_word', word(3000), 'assistant'), ('previous_word', word(3000), 'previous_user')],
            sparse_threshold=1.0,
        )
        text = self.text.fit_transform(pairs)
        manual, names = self._manual(pairs)
        self.feature_names_ = np.r_[self.text.get_feature_names_out(), np.array(names)]
        return sparse.hstack([text, sparse.csr_matrix(manual)], format='csr', dtype=np.float32)

    def transform(self, pairs):
        manual, _ = self._manual(pairs)
        return sparse.hstack([self.text.transform(pairs), sparse.csr_matrix(manual)], format='csr', dtype=np.float32)


def source_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('reaction_experiments.py').read_bytes()).hexdigest()


def data_fingerprint(dialogs, groups):
    digest = hashlib.sha256()
    for row, group in zip(dialogs.itertuples(index=False), groups):
        digest.update(json.dumps({'id': row.dialog_id, 'turns': row.turns, 'labels': row.labels, 'group': int(group)}, ensure_ascii=False, sort_keys=True).encode())
    return digest.hexdigest()


def bitset(values):
    return int.from_bytes(np.packbits(np.asarray(values, dtype=np.uint8), bitorder='little').tobytes(), 'little')


class FCAConceptRules:
    def __init__(self, n_attributes=64, rules_per_class=24):
        self.n_attributes = n_attributes
        self.rules_per_class = rules_per_class

    def fit(self, x, y, names):
        y = np.asarray(y)
        k = min(self.n_attributes, x.shape[1])
        self.selector = SelectKBest(chi2, k=k).fit(x, y)
        context = self.selector.transform(x).toarray().astype(bool)
        self.attribute_names_ = np.asarray(names)[self.selector.get_support()]
        n = len(y)
        all_objects = (1 << n) - 1
        attribute_extents = [bitset(context[:, j]) for j in range(k)]
        class_extents = [bitset(y == c) for c in CLASSES]
        self.prior_ = (np.array([(y == c).sum() for c in CLASSES]) + 1) / (n + len(CLASSES))
        min_support = max(3, min(20, int(0.005 * n)))
        min_class_support = max(2, min(5, int(0.002 * n)))
        concepts = {}

        def close_extent(extent):
            if extent in concepts or extent.bit_count() < min_support:
                return
            intent = tuple((j for j, bits in enumerate(attribute_extents) if extent & ~bits == 0))
            if not intent:
                return
            support = extent.bit_count()
            counts = np.array([(extent & c).bit_count() for c in class_extents])
            confidence = (counts + 1) / (support + len(CLASSES))
            lift = confidence / self.prior_
            concepts[extent] = {'intent': intent, 'support': support, 'counts': counts, 'confidence': confidence, 'lift': lift}
        close_extent(all_objects)
        for extent in attribute_extents:
            close_extent(extent)
        for a, b in combinations(attribute_extents, 2):
            close_extent(a & b)
        seeds = sorted(
            concepts,
            key=lambda e: float(np.max(concepts[e]['lift']) * np.log1p(concepts[e]['support'])),
            reverse=True,
        )[:64]
        for extent in seeds:
            for attribute in attribute_extents:
                close_extent(extent & attribute)
        self.concept_count_ = len(concepts)
        self.rules_ = []
        for class_index in range(len(CLASSES)):
            eligible = [(extent, c) for extent, c in concepts.items() if c['counts'][class_index] >= min_class_support and c['lift'][class_index] > 1.05]
            eligible.sort(
                key=lambda item: float(item[1]['lift'][class_index] * item[1]['confidence'][class_index] * np.log1p(item[1]['counts'][class_index])),
                reverse=True,
            )
            for extent, concept in eligible[:self.rules_per_class]:
                selected = np.all(context[:, concept['intent']], axis=1)
                assert bitset(selected) == extent
                assert tuple(np.flatnonzero(context[selected].all(axis=0))) == concept['intent']
                self.rules_.append({**concept, 'class_index': class_index})
        return self

    def transform(self, x):
        context = self.selector.transform(x).toarray().astype(bool)
        coverage = [context[:, r['intent']].all(axis=1) for r in self.rules_]
        output = []
        for k in range(len(CLASSES)):
            indices = [i for i, r in enumerate(self.rules_) if r['class_index'] == k]
            if not indices:
                output.extend([np.full(len(context), self.prior_[k]), np.full(len(context), self.prior_[k]), np.zeros(len(context)), np.zeros(len(context)), np.full(len(context), self.prior_[k])])
                continue
            hits = np.column_stack([coverage[i] for i in indices])
            confidence = np.array([self.rules_[i]['confidence'][k] for i in indices])
            log_lift = np.log(np.array([self.rules_[i]['lift'][k] for i in indices]))
            weight = np.sqrt(np.array([self.rules_[i]['support'] for i in indices]))
            counts = hits.sum(axis=1)
            covered = counts > 0
            maxima = np.where(covered, (hits * confidence).max(axis=1), self.prior_[k])
            means = np.where(covered, (hits * confidence).sum(axis=1) / np.maximum(counts, 1), self.prior_[k])
            weighted = np.where(
                covered,
                (hits * (confidence * weight)).sum(axis=1) / np.maximum((hits * weight).sum(axis=1), 1),
                self.prior_[k],
            )
            output.extend([maxima, means, (hits * log_lift).max(axis=1), counts / max(len(indices), 1), weighted])
        output.append(np.column_stack(coverage).any(axis=1).astype(float) if coverage else np.zeros(len(context)))
        return np.column_stack(output)

    def rule_table(self):
        return pd.DataFrame([{'class': CLASSES[r['class_index']], 'conditions': ' AND '.join(self.attribute_names_[list(r['intent'])]), 'support': r['support'], 'class_support': int(r['counts'][r['class_index']]), 'confidence': float(r['confidence'][r['class_index']]), 'lift': float(r['lift'][r['class_index']])} for r in self.rules_])


class BinarySVCFCA:
    def fit(self, pairs):
        self.encoder = BinaryInteractionEncoder()
        x = self.encoder.fit_transform(pairs)
        self.fca = FCAConceptRules().fit(x, pairs.reaction, self.encoder.feature_names_)
        self.svc_normalizer = Normalizer(copy=False)
        x = self.svc_normalizer.fit_transform(x)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.svc = LinearSVC(
                C=0.3,
                class_weight='balanced',
                dual=False,
                max_iter=5000,
                random_state=42,
            ).fit(x, pairs.reaction)
        self.svc_iterations_ = self.svc.n_iter_
        return self

    def transform(self, pairs):
        x = self.encoder.transform(pairs)
        rule_features = self.fca.transform(x)
        scores = self.svc.decision_function(self.svc_normalizer.transform(x))
        if scores.ndim != 2:
            raise ValueError('Training fold must contain all three reaction classes')
        order = [list(self.svc.classes_).index(c) for c in CLASSES]
        return (scores[:, order], rule_features)


def meta_input(scores, rules, variant):
    if variant == 'SVC only':
        return scores
    if variant == 'FCA only':
        return rules
    return np.column_stack([scores, rules])


def reaction_meta():
    return Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(C=1, max_iter=1500, random_state=42))])


def fit_stack(fit_dialogs, target_dialogs=None, inner_splits=3):
    pairs = make_pairs(fit_dialogs)
    scores = np.full((len(pairs), 3), np.nan)
    rules = np.full((len(pairs), 16), np.nan)
    inner = StratifiedGroupKFold(inner_splits, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(inner.split(fit_dialogs, fit_dialogs.reaction, fit_dialogs.group), start=1):
        assert not set(fit_dialogs.group.iloc[tr]) & set(fit_dialogs.group.iloc[va])
        training = make_pairs(fit_dialogs.iloc[tr].reset_index(drop=True))
        validation = make_pairs(fit_dialogs.iloc[va].reset_index(drop=True), labels=False)
        base = BinarySVCFCA().fit(training)
        sv, fc = base.transform(validation)
        mask = pairs.dialog_pos.isin(va).to_numpy()
        assert np.all(np.diff(va) > 0)
        scores[mask] = sv
        rules[mask] = fc
    assert np.isfinite(scores).all() and np.isfinite(rules).all()
    variants = ['SVC only', 'FCA only', 'SVC + FCA']
    models = {v: reaction_meta().fit(meta_input(scores, rules, v), pairs.reaction) for v in variants}
    base = BinarySVCFCA().fit(pairs)
    output = None
    target = None
    if target_dialogs is not None:
        target = make_pairs(target_dialogs, labels=False)
        sv, fc = base.transform(target)
        output = {v: m.predict_proba(meta_input(sv, fc, v))[:, [list(m.classes_).index(c) for c in CLASSES]] for v, m in models.items()}
    return (base, models, target, output)


def hard_dialog_labels(pairs, probabilities, n_dialogs):
    labels = CLASSES[probabilities.argmax(axis=1)]
    return np.array([last_non_neutral(labels[pairs.dialog_pos.to_numpy() == d].tolist()) for d in range(n_dialogs)])


def predict_fca_stack(bundle, dialogs, variant='SVC + FCA', aggregation='hard'):
    if len(dialogs) == 0:
        return np.empty(0, dtype=object)
    pairs = make_pairs(dialogs.reset_index(drop=True), labels=False)
    scores, rules = bundle['base'].transform(pairs)
    model = bundle['meta_models'][variant]
    probability = model.predict_proba(meta_input(scores, rules, variant))[:, [list(model.classes_).index(c) for c in CLASSES]]
    return CLASSES[soft_last_non_neutral(pairs, probability, len(dialogs)).argmax(axis=1)] if aggregation == 'soft' else hard_dialog_labels(pairs, probability, len(dialogs))

@threadpool_limits.wrap(limits=1)


def run_experiment(dialogs, groups, n_splits=5, inner_splits=3, train_final=True):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    dialogs = dialogs.reset_index(drop=True).copy()
    dialogs['group'] = np.asarray(groups)
    y = dialogs.reaction.to_numpy()
    all_pairs = make_pairs(dialogs)
    variants = ['SVC only', 'FCA only', 'SVC + FCA']
    probabilities = {v: np.full((len(all_pairs), 3), np.nan) for v in variants}
    predictions = {v + ' / ' + mode: np.full(len(y), '', dtype=object) for v in variants for mode in ['hard', 'soft']}
    fold_scores = {name: [] for name in predictions}
    rule_tables = []
    outer = StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(outer.split(dialogs, y, groups), start=1):
        tick = time.perf_counter()
        assert not set(np.asarray(groups)[tr]) & set(np.asarray(groups)[va])
        base, models, target, pred = fit_stack(
            dialogs.iloc[tr].reset_index(drop=True),
            dialogs.iloc[va].reset_index(drop=True),
            inner_splits,
        )
        rule_table = base.fca.rule_table()
        rule_table['fold'] = fold
        rule_tables.append(rule_table)
        for variant, p in pred.items():
            assert np.isfinite(p).all()
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-06)
            probabilities[variant][all_pairs.dialog_pos.isin(va).to_numpy()] = p
            predictions[variant + ' / hard'][va] = hard_dialog_labels(target, p, len(va))
            predictions[variant + ' / soft'][va] = CLASSES[soft_last_non_neutral(target, p, len(va)).argmax(axis=1)]
        for name, prediction in predictions.items():
            score = f1_score(y[va], prediction[va], average='macro', zero_division=0)
            fold_scores[name].append(float(score))
            print(f'  {name}: {score:.4f}')
        joblib.dump(
            {'completed_folds': fold, 'predictions': predictions, 'probabilities': probabilities, 'fold_scores': fold_scores, 'dialog_ids': dialogs.dialog_id.tolist()},
            RESULT_DIR / 'checkpoint.joblib',
        )
    table = pd.DataFrame([{'Method': name, **metrics(y, pred), 'Mean CV Macro-F1': np.mean(fold_scores[name]), 'Std CV Macro-F1': np.std(fold_scores[name])} for name, pred in predictions.items()]).set_index('Method')
    pair_table = pd.DataFrame([{'Method': name, **metrics(all_pairs.reaction.to_numpy(), CLASSES[p.argmax(axis=1)])} for name, p in probabilities.items()]).set_index('Method')
    mixed = presence_labels(dialogs).all(axis=1)
    mixed_table = pd.DataFrame([{'Method': name, 'Mixed accuracy': np.mean(y[mixed] == pred[mixed])} for name, pred in predictions.items()]).set_index('Method')
    reports = {name: classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0) for name, pred in predictions.items()}
    table.to_csv(RESULT_DIR / 'dialog_metrics.csv')
    pair_table.to_csv(RESULT_DIR / 'pair_metrics.csv')
    mixed_table.to_csv(RESULT_DIR / 'mixed_metrics.csv')
    pd.concat(rule_tables, ignore_index=True).to_csv(RESULT_DIR / 'rules.csv', index=False)
    bundle = None
    if train_final:
        base, models, _, _ = fit_stack(dialogs, inner_splits=inner_splits)
        bundle = {
            'base': base,
            'meta_models': models,
            'classes': CLASSES,
            'train_dialog_ids': dialogs.dialog_id.tolist(),
        }
        joblib.dump(bundle, RESULT_DIR / 'final_model.joblib')
        base.fca.rule_table().to_csv(RESULT_DIR / 'final_rules.csv', index=False)
    summary = {
        'seconds': time.perf_counter() - start,
        'reports': reports,
        'fold_scores': fold_scores,
        'dialog_ids': dialogs.dialog_id.tolist(),
        'source_sha256': source_fingerprint(),
        'data_sha256': data_fingerprint(dialogs, groups),
        'outer_splits': n_splits,
        'inner_splits': inner_splits,
        'final_trained': train_final,
        'fca_search': '64 selected binary attributes; closed singleton/pair concepts plus 64 seed extensions; 24 rules per class',
    }
    (RESULT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\nFCA, реплики:\n' + pair_table.round(4).to_string())
    print('\nFCA, диалоги:\n' + table.round(4).to_string())
    print('\nДиалоги с обоими знаками:\n' + mixed_table.round(4).to_string())
    return (table, pair_table, mixed_table, bundle)
