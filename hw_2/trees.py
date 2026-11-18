"""Новые модели HW2: деревья и ансамбли на репликах, стекинг на OOF.

Объект — реплика пользователя j >= 1. Признаки (учатся на обучающей части):
  raw  — TF-IDF: user слова 1-2 (12k), user символы 3-5 (18k),
         ответ ассистента до реплики (6k, x0.4), ответ после (6k, x0.4);
  turn — SVD(100) от raw + 37 числовых (структура, длины, регулярки, пересечения).
Метка диалога — по последней реплике с max(P(neg), P(pos)) >= порога, иначе neutral.
"""

import hashlib
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import old_models as om
import reaction_models as rm

SEED = 42
CLASSES = np.array(["negative", "neutral", "positive"])
STRUCTURE = ["position", "log_length", "n_users", "questions", "exclamations"]
SOURCES = ["user", "assistant", "following"]
PATTERN_NAMES = ["negation", "disagree", "fix", "explain", "thanks", "praise",
                 "apology", "admit_mistake", "welcome"]
NUMERIC_NAMES = (STRUCTURE + ["log_assistant_len", "log_following_len"]
                 + [f"{s}_{p}" for s in SOURCES for p in PATTERN_NAMES]
                 + ["overlap_user_assistant", "overlap_user_following", "overlap_assistant_following"])


# --- признаки реплик --------------------------------------------------------

def turn_pairs(df, training=False):
    rows = []
    for d, (turns, labels) in enumerate(zip(df["turns"], df["labels"] if training else [None] * len(df))):
        users = [i for i, t in enumerate(turns) if t["role"] == "user"]
        for j, i in enumerate(users[1:], 1):
            prev = turns[i - 1]["text"] if turns[i - 1]["role"] == "assistant" else ""
            nxt = turns[i + 1]["text"] if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant" else ""
            text = turns[i]["text"]
            rows.append(dict(
                dialog_pos=d, user=text, assistant=prev, following=nxt[:600],
                position=j / max(len(users) - 1, 1), log_length=np.log1p(len(text)), n_users=len(users),
                questions=text.count("?"), exclamations=text.count("!"),
                log_assistant_len=np.log1p(len(prev)), log_following_len=np.log1p(len(nxt)),
                reaction=labels["user_turns"][j]["reaction"] if training else None))
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


PAIR_COLUMNS = ["dialog_pos", "user", "assistant", "following", *STRUCTURE,
                "log_assistant_len", "log_following_len", "reaction"]


def numeric(p):
    cols = [p[STRUCTURE + ["log_assistant_len", "log_following_len"]].to_numpy(np.float32)]
    words = {}
    for s in SOURCES:
        text = p[s].str.lower()
        cols += [text.str.contains(pattern).to_numpy(np.float32)[:, None] for pattern in rm.PATTERNS]
        words[s] = [set(re.findall(r"\b\w+\b", t)) for t in text]
    for a, b in [("user", "assistant"), ("user", "following"), ("assistant", "following")]:
        cols.append(np.array([[len(x & y) / max(len(x | y), 1)] for x, y in zip(words[a], words[b])], np.float32))
    return np.hstack(cols)


def tfidf(n, **kw):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=n, sublinear_tf=True, dtype=np.float32, **kw)


class TurnFeatures:
    def fit(self, df):
        p = turn_pairs(df, training=True)
        self.tfidf = ColumnTransformer(
            [("user_word", tfidf(12000), "user"),
             ("user_char", tfidf(18000, analyzer="char_wb").set_params(ngram_range=(3, 5), min_df=3), "user"),
             ("assistant", tfidf(6000), "assistant"),
             ("following", tfidf(6000), "following")],
            transformer_weights={"user_word": 1, "user_char": 1, "assistant": 0.4, "following": 0.4})
        raw = sparse.csr_matrix(self.tfidf.fit_transform(p))
        self.svd = TruncatedSVD(100, random_state=SEED).fit(raw)
        self.train = self._assemble(p, raw)
        return self

    def transform(self, df):
        p = turn_pairs(df)
        return self._assemble(p, sparse.csr_matrix(self.tfidf.transform(p)))

    def __getstate__(self):  # в файл модели не сохраняем матрицы обучающей выборки
        return {k: v for k, v in self.__dict__.items() if k != "train"}

    def _assemble(self, p, raw):
        return {"pairs": p, "raw": raw, "turn": np.hstack([self.svd.transform(raw), numeric(p)])}

    def names(self, kind):
        if kind == "raw":
            return list(self.tfidf.get_feature_names_out())
        return [f"svd_{i}" for i in range(100)] + NUMERIC_NAMES


_cache = {}


def features(df):
    """TurnFeatures, обученные на df; кэш по набору диалогов (несколько моделей на одном фолде)."""
    key = hashlib.sha1("".join(df["dialog_id"]).encode()).hexdigest()
    if key not in _cache:
        if len(_cache) >= 8:
            _cache.pop(next(iter(_cache)))
        _cache[key] = TurnFeatures().fit(df)
    return _cache[key]


def svm_base(df):
    """Два LinearSVC hw1_decision_2x, обученные на df."""
    return om.fit_base(om.training_pairs(df), om._load("hw1_decision_2x"), (0.0, 0.5), (2.0, 2.0, 2.0))


def svm_scores(base, df):
    s, _, _ = rm._scores(base, rm._pairs(om._dialogs(df)), need_nb=False)
    return np.column_stack([s["positive"], s["negative"]])


def oof_svm_scores(df):
    """Оценки SVM для реплик df out-of-fold: 3 внутренних фолда по группам."""
    pos = turn_pairs(df)["dialog_pos"].to_numpy()
    out = np.zeros((len(pos), 2))
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
    for tr, va in cv.split(df, df["reaction"], df["group"]):
        out[np.isin(pos, va)] = svm_scores(svm_base(df.iloc[tr]), df.iloc[va])
    return out


# --- от реплик к диалогу ----------------------------------------------------

def last_reaction(dialog_pos, prob, n, threshold=0.5):
    """Метки диалогов и индекс выбранной реплики (-1, если её нет)."""
    hit = np.maximum(prob[:, 0], prob[:, 2]) >= threshold
    last = np.full(n, -1)
    np.maximum.at(last, dialog_pos[hit], np.flatnonzero(hit))
    labels = np.full(n, "neutral", dtype=object)
    s = last[last >= 0]
    labels[last >= 0] = np.where(prob[s, 2] > prob[s, 0], "positive", "negative")
    return labels, last


def dialog_outputs(dialog_pos, prob, n, threshold=0.5):
    """Вход для стекинга: P выбранной реплики (3 числа), max P(neg), max P(pos)."""
    _, last = last_reaction(dialog_pos, prob, n, threshold)
    out = np.zeros((n, 5))
    out[:, 1] = 1
    out[last >= 0, :3] = prob[last[last >= 0]]
    np.maximum.at(out[:, 3], dialog_pos, prob[:, 0])
    np.maximum.at(out[:, 4], dialog_pos, prob[:, 2])
    return out


def proba(model, x):
    return model.predict_proba(x)[:, [list(model.classes_).index(c) for c in CLASSES]]


# --- модели -----------------------------------------------------------------

class TurnModel:
    """Классификатор реплик + правило последней реакции.
    kind: "turn" (137 плотных) или "raw" (TF-IDF); svm: добавить OOF-оценки SVM HW1."""

    def __init__(self, estimator, kind="turn", threshold=0.5, balanced=True, svm=False):
        self.estimator, self.kind, self.threshold = estimator, kind, threshold
        self.balanced, self.svm = balanced, svm

    def _x(self, f, scores):
        x = f[self.kind]
        if not self.svm:
            return x
        return sparse.hstack([x, scores], format="csr") if sparse.issparse(x) else np.hstack([x, scores])

    def fit(self, df):
        self.rep = features(df)
        f = self.rep.train
        y = f["pairs"]["reaction"].to_numpy()
        w = compute_sample_weight("balanced", y) if self.balanced else None
        scores = None
        if self.svm:
            scores, self.base = oof_svm_scores(df), svm_base(df)
        self.model = clone(self.estimator).fit(self._x(f, scores), y, sample_weight=w)
        return self

    def predict_with_proba(self, df):
        if not len(turn_pairs(df)):  # нет реплик j >= 1 -> neutral
            self.turn_output_ = (np.zeros(0, int), np.zeros((0, 3)))
            return np.full(len(df), "neutral", dtype=object), dialog_outputs(*self.turn_output_, len(df))
        f = self.rep.transform(df)
        pos = f["pairs"]["dialog_pos"].to_numpy()
        prob = proba(self.model, self._x(f, svm_scores(self.base, df) if self.svm else None))
        self.turn_output_ = (pos, prob)
        return last_reaction(pos, prob, len(df), self.threshold)[0], dialog_outputs(pos, prob, len(df), self.threshold)

    def predict(self, df):
        return self.predict_with_proba(df)[0]

    def predict_dialogs(self, dialogs):
        return self.predict(pd.DataFrame(dialogs))


class DialogTextModel:
    """Модель на TF-IDF всего текста диалога (без разбиения на реплики)."""

    def __init__(self, estimator):
        self.pipeline = make_pipeline(tfidf(20000), estimator)

    def fit(self, df):
        y = df["reaction"].to_numpy()
        texts = [om.reaction_dialog_text(d) for d in om._dialogs(df)]
        self.pipeline.fit(texts, y, **{self.pipeline.steps[-1][0] + "__sample_weight": compute_sample_weight("balanced", y)})
        return self

    def predict(self, df):
        return self.pipeline.predict([om.reaction_dialog_text(d) for d in om._dialogs(df)])


class OOFStack:
    """Мета-LR на выходах базовых моделей. Выходы для обучения меты — out-of-fold
    (3 внутренних фолда по группам), затем базы переобучаются на всей части."""

    def __init__(self, bases, C=1.0, balanced=True):
        self.bases, self.C, self.balanced = bases, C, balanced

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "bases"}

    def fit(self, df):
        df = df.reset_index(drop=True)
        y = df["reaction"].to_numpy()
        oof = {name: [None] * len(df) for name in self.bases}
        cv = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
        for tr, va in cv.split(df, y, df["group"]):
            for name, make in self.bases.items():
                out = make().fit(df.iloc[tr].reset_index(drop=True)).predict_with_proba(df.iloc[va].reset_index(drop=True))[1]
                for i, row in zip(va, out):
                    oof[name][i] = row
        x = np.hstack([np.vstack(oof[name]) for name in self.bases])
        self.models = {name: make().fit(df) for name, make in self.bases.items()}
        self.meta = make_pipeline(StandardScaler(), LogisticRegression(
            C=self.C, class_weight="balanced" if self.balanced else None, max_iter=2000)).fit(x, y)
        return self

    def predict_with_proba(self, df):
        x = np.hstack([m.predict_with_proba(df)[1] for m in self.models.values()])
        p = proba(self.meta, x)
        return CLASSES[p.argmax(1)], p

    def predict(self, df):
        return self.predict_with_proba(df)[0]


class BaggedDecision2x:
    """Бэггинг Decision_2x: n копий базы HW1 на бутстрепе диалогов,
    оценки двух SVM усредняются, дальше те же пороги и правило последней реакции."""

    def __init__(self, n=10, thresholds=(0.0, 0.5)):
        self.n, self.thresholds = n, thresholds

    def fit(self, df):
        rng = np.random.RandomState(SEED)
        template = om._load("hw1_decision_2x")
        self.bases = []
        for _ in range(self.n):
            sample = df.iloc[rng.randint(0, len(df), len(df))].reset_index(drop=True)
            self.bases.append(om.fit_base(om.training_pairs(sample), template, self.thresholds, (2.0, 2.0, 2.0)))
        return self

    def predict_with_proba(self, df):
        pairs = rm._pairs(om._dialogs(df))
        s = {label: np.mean([rm._scores(b, pairs, need_nb=False)[0][label] for b in self.bases], axis=0)
             for label in ["positive", "negative"]}
        labels = rm._decision(pairs, s, self.thresholds, len(df))
        ids = pairs["dialog_pos"].to_numpy(dtype=int)
        out = np.full((len(df), 2), -3.0)
        np.maximum.at(out[:, 0], ids, s["positive"])
        np.maximum.at(out[:, 1], ids, s["negative"])
        return labels, out

    def predict(self, df):
        return self.predict_with_proba(df)[0]
