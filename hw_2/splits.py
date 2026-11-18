"""Данные и фолды HW2.

Схема одна для всех моделей, старых и новых:
StratifiedGroupKFold(5, shuffle=True, random_state=42) по группам
почти-дубликатов на всех 3000 размеченных диалогах. Отложенной выборки нет.
Так же устроен ноутбук коллег (reference/trees_context_full_colab.ipynb),
поэтому его числа сопоставимы с нашими.

Группы, как в HW1: TF-IDF реплик пользователя, ближайший сосед по косинусу,
сходство >= 0.95 -> одна группа (union-find). Получается 2987 групп.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
DEFAULT_TRAIN = HERE / "data" / "hw1_train.jsonl"

SEED = 42
N_SPLITS = 5
SIMILARITY = 0.95


def load_labeled(path=DEFAULT_TRAIN) -> pd.DataFrame:
    """Диалоги с меткой реакции, в порядке файла, индекс 0..n-1."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            labels = record.get("labels") or {}
            if labels.get("reaction") is None:
                continue
            rows.append({
                "dialog_id": record["dialog_id"],
                "turns": record["turns"],
                "labels": labels,
                "reaction": labels["reaction"],
            })
    return pd.DataFrame(rows)


def make_groups(df: pd.DataFrame) -> np.ndarray:
    """Номер группы почти-дубликатов, как в HW1 (корень union-find)."""
    user_text = df["turns"].apply(
        lambda turns: "\n".join(t["text"] for t in turns if t["role"] == "user")
    )
    x = TfidfVectorizer(max_features=10000, stop_words="english").fit_transform(user_text)
    knn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute").fit(x)
    distances, indices = knn.kneighbors(x)
    similarities = 1 - distances[:, 1]

    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(df)):
        j = indices[i, 1]
        if i != j and similarities[i] >= SIMILARITY:
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[rb] = ra
    return np.array([find(i) for i in range(len(df))])


def make_cv(n_splits=N_SPLITS, seed=SEED):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def assign_folds(df: pd.DataFrame, n_splits=N_SPLITS, seed=SEED) -> np.ndarray:
    """Номер валидационного фолда для каждого диалога (по колонкам reaction, group)."""
    fold = np.full(len(df), -1)
    cv = make_cv(n_splits, seed)
    for k, (_, va) in enumerate(cv.split(df, df["reaction"], df["group"])):
        fold[va] = k
    assert (fold >= 0).all()
    return fold


def load_data(path=DEFAULT_TRAIN) -> pd.DataFrame:
    """3000 диалогов с колонками group и fold."""
    df = load_labeled(path)
    df["group"] = make_groups(df)
    df["fold"] = assign_folds(df)
    check_folds(df)
    return df


def iter_folds(df: pd.DataFrame):
    """(номер фолда, индексы fit, индексы valid) по колонке fold."""
    fold = df["fold"].to_numpy()
    for k in range(int(fold.max()) + 1):
        yield k, np.flatnonzero(fold != k), np.flatnonzero(fold == k)


def check_folds(df: pd.DataFrame) -> None:
    assert len(df) == 3000, len(df)
    assert df["group"].nunique() == 2987, df["group"].nunique()
    for k, fit, val in iter_folds(df):
        assert not set(df["group"].iloc[fit]) & set(df["group"].iloc[val]), f"фолд {k}: группы пересекаются"
