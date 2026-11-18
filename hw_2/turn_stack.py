"""Стекинг на уровне реплик (лучшая новая модель).

Базы: два LinearSVC из hw1_decision_2x (старая модель) и деревья на репликах.
Вход меты для реплики: 2 оценки SVM + по 3 вероятности от каждого дерева.
Для обучения меты эти выходы out-of-fold (3 внутренних фолда по группам).
Метка диалога — правило последней реакции; порог подбирается по OOF-вероятностям
меты на обучающей части (5 фолдов по диалогам), валидация в подборе не участвует.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict

import trees as T
from cv import macro_f1
from splits import iter_folds

SEED = 42
GRID = np.arange(0.3, 0.71, 0.025)


def base_outputs(svm, tree_models, df):
    """Матрица реплики x (2 + 3 * число деревьев)."""
    cols = [T.svm_scores(svm, df)]
    for model in tree_models.values():
        model.predict_with_proba(df)
        cols.append(model.turn_output_[1])
    return np.hstack(cols)


def fit_bases(df, trees):
    return T.svm_base(df), {name: make().fit(df) for name, make in trees.items()}


def oof_outputs(df, trees, inner=3):
    pos = T.turn_pairs(df)["dialog_pos"].to_numpy()
    x = None
    cv = StratifiedGroupKFold(inner, shuffle=True, random_state=SEED)
    for tr, va in cv.split(df, df["reaction"], df["group"]):
        train, test = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
        block = base_outputs(*fit_bases(train, trees), test)
        if x is None:
            x = np.zeros((len(pos), block.shape[1]))
        x[np.isin(pos, va)] = block
    return x


def tune_threshold(meta, x, y_turn, pos, y_dialog):
    p = cross_val_predict(meta, x, y_turn, groups=pos, cv=GroupKFold(5), method="predict_proba")
    scores = [macro_f1(y_dialog, T.last_reaction(pos, p, len(y_dialog), t)[0]) for t in GRID]
    return GRID[int(np.argmax(scores))]


class TurnStack:
    def __init__(self, trees, meta, weights=(1, 1, 1)):
        self.trees, self.meta, self.weights = trees, meta, weights

    def fit(self, df):
        df = df.reset_index(drop=True)
        pairs = T.turn_pairs(df, training=True)
        pos, y_turn = pairs["dialog_pos"].to_numpy(), pairs["reaction"].to_numpy()
        x = oof_outputs(df, self.trees)
        self.threshold = tune_threshold(clone(self.meta), x, y_turn, pos, df["reaction"].to_numpy())
        self.meta_ = clone(self.meta).fit(x, y_turn)
        self.svm_, self.tree_models_ = fit_bases(df, self.trees)
        return self

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "trees"}

    def predict_with_proba(self, df):
        pos = T.turn_pairs(df)["dialog_pos"].to_numpy()
        if not len(pos):
            return np.full(len(df), "neutral", dtype=object), np.zeros((len(df), 5))
        prob = T.proba(self.meta_, base_outputs(self.svm_, self.tree_models_, df)) * np.asarray(self.weights)
        self.turn_output_ = (pos, prob)
        return T.last_reaction(pos, prob, len(df), self.threshold)[0], T.dialog_outputs(pos, prob, len(df), self.threshold)

    def predict(self, df):
        return self.predict_with_proba(df)[0]

    def predict_dialogs(self, dialogs):
        return self.predict(pd.DataFrame(dialogs))


# --- быстрый подбор меты: выходы баз считаются один раз ---------------------

PATH = "artifacts/level1_turns.joblib"


def level1_turns(df, trees):
    """Для каждого внешнего фолда: OOF-выходы баз на обучающей части и выходы на валидации."""
    if os.path.exists(PATH):
        return joblib.load(PATH)
    out = {}
    for k, fit_idx, val_idx in iter_folds(df):
        fit, val = df.iloc[fit_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
        pairs = T.turn_pairs(fit, training=True)
        out[k] = dict(x_fit=oof_outputs(fit, trees), y_fit=pairs["reaction"].to_numpy(),
                      pos_fit=pairs["dialog_pos"].to_numpy(), y_dialog_fit=fit["reaction"].to_numpy(),
                      x_val=base_outputs(*fit_bases(fit, trees), val),
                      pos_val=T.turn_pairs(val)["dialog_pos"].to_numpy())
    joblib.dump(out, PATH)
    return out


def oof_predict(df, level1, meta, columns=slice(None), weights=(1, 1, 1)):
    """Предсказания по внешним фолдам; weights — множители вероятностей классов реплик."""
    pred, thresholds = np.empty(len(df), dtype=object), []
    for k, _, val_idx in iter_folds(df):
        d = level1[k]
        x_fit, x_val = d["x_fit"][:, columns], d["x_val"][:, columns]
        t = tune_threshold(clone(meta), x_fit, d["y_fit"], d["pos_fit"], d["y_dialog_fit"])
        prob = T.proba(clone(meta).fit(x_fit, d["y_fit"]), x_val) * np.asarray(weights)
        pred[val_idx] = T.last_reaction(d["pos_val"], prob, len(val_idx), t)[0]
        thresholds.append(t)
    return pred, thresholds
