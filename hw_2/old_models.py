"""Модели HW1, переобучаемые на фолдах HW2 с гиперпараметрами HW1.

Гиперпараметры не подбираются заново: берём их из сохранённых
artifacts/hw1_*.joblib (clone копирует параметры без обученных весов).
Предсказание идёт через reaction_models — тот же код, что в solution.ipynb,
поэтому модель, обученная здесь на всех 3000, совпадает с сохранённой
(проверка в experiments.ipynb, раздел 2).

Модели:
- TextModel: hw1_naive_bayes / hw1_logistic_regression / hw1_svm —
  TF-IDF текста всего диалога + линейная модель;
- Decision2x: hw1_decision_2x и hw1_grisha_909 — по репликам, два LinearSVC
  (positive и negative против остальных), правило последней реакции;
- ScoreStack: hw1_score_stacking_lr и hw1_stack_909 — LR по 12 признакам
  диалога из оценок Decision2x и ComplementNB.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from threadpoolctl import threadpool_limits

import reaction_models as rm
from solution_utils import reaction_dialog_text

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
SEED = 42


def _load(name):
    return joblib.load(ARTIFACTS / f"{name}.joblib")


def _dialogs(df: pd.DataFrame) -> list[dict]:
    return [{"dialog_id": d, "turns": t} for d, t in zip(df["dialog_id"], df["turns"])]


# ---------------------------------------------------------------------------
#  hw1_naive_bayes, hw1_logistic_regression, hw1_svm
# ---------------------------------------------------------------------------

class TextModel:
    """Пайплайн sklearn на тексте всего диалога («role: text», clean_text)."""

    def __init__(self, artifact: str):
        self.pipeline = clone(_load(artifact))

    def fit(self, df):
        texts = [reaction_dialog_text(d) for d in _dialogs(df)]
        self.pipeline.fit(texts, df["reaction"].to_numpy())
        return self

    def predict(self, df):
        texts = [reaction_dialog_text(d) for d in _dialogs(df)]
        return self.pipeline.predict(texts)

    def predict_with_proba(self, df):
        """Метки и оценки классов (вход для стекинга)."""
        texts = [reaction_dialog_text(d) for d in _dialogs(df)]
        model = self.pipeline
        scores = model.predict_proba(texts) if hasattr(model, "predict_proba") else model.decision_function(texts)
        return model.predict(texts), scores


# ---------------------------------------------------------------------------
#  hw1_decision_2x и hw1_grisha_909
# ---------------------------------------------------------------------------

def training_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Реплики пользователя j >= 1 (как reaction_models._pairs) + их метка."""
    pairs = rm._pairs(_dialogs(df))
    reactions = [
        labels["user_turns"][j]["reaction"]
        for labels, turns in zip(df["labels"], df["turns"])
        for j in range(1, sum(t["role"] == "user" for t in turns))
    ]
    assert len(reactions) == len(pairs)
    pairs["reaction"] = reactions
    return pairs


def sample_weights(pairs: pd.DataFrame, policy=(2.0, 2.0, 2.0)) -> np.ndarray:
    """Веса реплик из HW1 (experiments/course_decision_weight.py).

    policy = (решающая реплика, поздние нейтральные после нее,
              реплики полностью нейтральных диалогов); остальные — 1.
    Решающая реплика — последняя ненейтральная: по ней ставится метка диалога.
    """
    y = pairs["reaction"].to_numpy()
    ids = pairs["dialog_pos"].to_numpy(dtype=int)
    weights = np.ones(len(pairs))
    for dialog in np.unique(ids):
        idx = np.flatnonzero(ids == dialog)
        nonneutral = idx[y[idx] != "neutral"]
        if len(nonneutral):
            decisive = nonneutral[-1]
            weights[decisive] = policy[0]
            weights[idx[(idx > decisive) & (y[idx] == "neutral")]] = policy[1]
        else:
            weights[idx] = policy[2]
    return weights / weights.mean()


def fit_base(pairs: pd.DataFrame, template: dict, thresholds, policy) -> dict:
    """Обучить базу Decision2x на репликах. Возвращает словарь в формате
    artifacts/hw1_decision_2x.joblib (его понимает reaction_models)."""
    parts, blocks = [], []
    for part in template["parts"]:
        transformer = clone(part["transformer"])
        block = transformer.fit_transform(pairs[part["columns"]]) * part["weight"]
        blocks.append(sparse.csr_matrix(block))
        parts.append({**part, "transformer": transformer})
    signal_scaler = clone(template["signal_scaler"])
    signals = rm._signals(pairs)
    blocks.append(signal_scaler.fit_transform(signals) * template["signal_weight"])
    x = sparse.hstack(blocks, format="csr")

    weights = sample_weights(pairs, policy)
    y = pairs["reaction"].to_numpy()
    models = {}
    for label, (svc, _) in template["models"].items():
        svc = clone(svc).fit(x, y == label, sample_weight=weights)
        # оценки делим на их разброс на train, чтобы пороги были в одной шкале
        scale = max(float(np.std(svc.decision_function(x))), 1e-6)
        models[label] = (svc, scale)
    return {
        "parts": parts,
        "signal_scaler": signal_scaler,
        "signal_weight": template["signal_weight"],
        "models": models,
        "thresholds": tuple(thresholds),
    }


def balanced_subsample(df: pd.DataFrame, seed=SEED) -> pd.DataFrame:
    """По n диалогов каждого класса, n = размер наименьшего класса (для 909)."""
    n = df["reaction"].value_counts().min()
    parts = [g.sample(n=n, random_state=seed) for _, g in df.groupby("reaction")]
    return pd.concat(parts).sort_index().reset_index(drop=True)


class Decision2x:
    """hw1_decision_2x: пороги (0, 0.5), веса (2,2,2), все диалоги фолда.
    hw1_grisha_909: пороги (0, 0), подвыборка 1:1:1 (balanced=True)."""

    def __init__(self, artifact="hw1_decision_2x", thresholds=(0.0, 0.5),
                 policy=(2.0, 2.0, 2.0), balanced=False):
        self.artifact = artifact
        self.thresholds = thresholds
        self.policy = policy
        self.balanced = balanced

    def fit(self, df):
        if self.balanced:
            df = balanced_subsample(df)
        with threadpool_limits(limits=1):
            self.base = fit_base(training_pairs(df), _load(self.artifact),
                                 self.thresholds, self.policy)
        return self

    def predict(self, df):
        return np.asarray(rm.Decision2xClassifier(self.base).predict(_dialogs(df)))

    def predict_with_proba(self, df):
        """Метки и 4 числа на диалог для стекинга: оценки pos и neg решающей
        реплики (0, если её нет) и максимумы оценок по репликам."""
        pairs = rm._pairs(_dialogs(df))
        s, _, _ = rm._scores(self.base, pairs, need_nb=False)
        labels = rm._decision(pairs, s, self.thresholds, len(df))
        ids = pairs["dialog_pos"].to_numpy(dtype=int)
        hit = (s["positive"] >= self.thresholds[0]) | (s["negative"] >= self.thresholds[1])
        last = np.full(len(df), -1)
        np.maximum.at(last, ids[hit], np.flatnonzero(hit))
        out = np.full((len(df), 4), -3.0)
        out[last >= 0, 0] = s["positive"][last[last >= 0]]
        out[last >= 0, 1] = s["negative"][last[last >= 0]]
        np.maximum.at(out[:, 2], ids, s["positive"])
        np.maximum.at(out[:, 3], ids, s["negative"])
        return labels, out


# ---------------------------------------------------------------------------
#  hw1_score_stacking_lr и hw1_stack_909
# ---------------------------------------------------------------------------

def _nb_inputs(base, pairs):
    """Оценки SVM, сигналы и TF-IDF-блоки для NB (без structure)."""
    return rm._scores(base, pairs, need_nb=True)


def _fit_nb(nb_x, pairs):
    """ComplementNB: ненейтральная реплика (1) против нейтральной (0)."""
    return ComplementNB().fit(nb_x, (pairs["reaction"].to_numpy() != "neutral").astype(int))


def _nb_odds(nb, nb_x):
    logp = nb.predict_log_proba(nb_x)
    classes = list(nb.classes_)
    return logp[:, classes.index(1)] - logp[:, classes.index(0)]


class ScoreStack:
    """База Decision2x + ComplementNB -> 12 признаков диалога -> StandardScaler + LR.

    Признаки для обучения мета-модели — out-of-fold: внутренняя CV из 3 фолдов
    по группам внутри обучающей части, база и NB учатся на 2/3 и оценивают 1/3.
    Иначе SVM, видевший свои реплики, дает завышенные оценки и мета-модель
    учится не на том распределении, что будет на тесте.
    """

    def __init__(self, base_model: Decision2x, meta_artifact="hw1_score_stacking_lr", inner_splits=3):
        self.base_model = base_model
        self.meta_artifact = meta_artifact
        self.inner_splits = inner_splits

    def _features(self, base, nb, df, pairs):
        scores, signals, nb_x = _nb_inputs(base, pairs)
        odds = _nb_odds(nb, nb_x)
        return rm._stack_features(pairs, scores, signals, odds, base["thresholds"], len(df))

    def fit(self, df):
        bm = self.base_model
        if bm.balanced:
            df = balanced_subsample(df)
        df = df.reset_index(drop=True)
        template = _load(bm.artifact)
        y = df["reaction"].to_numpy()

        with threadpool_limits(limits=1):
            # 1. OOF-признаки диалогов для мета-модели
            meta_x = np.zeros((len(df), 12))
            cv = StratifiedGroupKFold(self.inner_splits, shuffle=True, random_state=SEED)
            for tr, va in cv.split(df, y, df["group"]):
                inner_train = df.iloc[tr].reset_index(drop=True)
                inner_valid = df.iloc[va].reset_index(drop=True)
                pt = training_pairs(inner_train)
                base = fit_base(pt, template, bm.thresholds, bm.policy)
                nb = _fit_nb(_nb_inputs(base, pt)[2], pt)
                pv = rm._pairs(_dialogs(inner_valid))
                meta_x[va] = self._features(base, nb, inner_valid, pv)

            # 2. база и NB на всей обучающей части — их увидит тест
            pairs = training_pairs(df)
            self.base = fit_base(pairs, template, bm.thresholds, bm.policy)
            nb = _fit_nb(_nb_inputs(self.base, pairs)[2], pairs)

            # 3. мета-модель
            saved = _load(self.meta_artifact)
            scaler = clone(saved["scaler"])
            meta = clone(saved["meta"]).fit(scaler.fit_transform(meta_x), y)
        self.stack = {"nb": nb, "scaler": scaler, "meta": meta}
        return self

    def predict(self, df):
        model = rm.ScoreStackingLRClassifier(self.base, self.stack)
        return np.asarray(model.predict(_dialogs(df)))


# ---------------------------------------------------------------------------
#  Все старые модели с настройками HW1
# ---------------------------------------------------------------------------

OLD_MODELS = {
    "hw1_naive_bayes": lambda: TextModel("hw1_naive_bayes"),
    "hw1_logistic_regression": lambda: TextModel("hw1_logistic_regression"),
    "hw1_svm": lambda: TextModel("hw1_svm"),
    "hw1_decision_2x": lambda: Decision2x(),
    "hw1_score_stacking_lr": lambda: ScoreStack(Decision2x()),
    "hw1_grisha_909": lambda: Decision2x("hw1_grisha_909", thresholds=(0.0, 0.0), balanced=True),
    "hw1_stack_909": lambda: ScoreStack(
        Decision2x("hw1_grisha_909", thresholds=(0.0, 0.0), balanced=True),
        meta_artifact="hw1_stack_909"),
}
