"""Общая кросс-валидация для всех моделей HW2.

Модель — любой объект с методами fit(df) и predict(df), где df — таблица
диалогов (колонки turns, labels, reaction), predict возвращает метки диалогов.
Если есть predict_with_proba(df) -> (метки, вероятности), сохраняются
и вероятности (нужны для стекинга и подбора порога).

run_cv обучает новую модель на каждом фолде, замеряет время обучения
и предсказания, сохраняет OOF и результат в artifacts/cv/<name>.joblib.
Повторный вызов с тем же именем читает кэш (force=True пересчитывает).
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

from splits import iter_folds

HERE = Path(__file__).resolve().parent
CV_DIR = HERE / "artifacts" / "cv"
REACTIONS = ["negative", "neutral", "positive"]


def macro_f1(y_true, y_pred, sample_weight=None) -> float:
    return f1_score(y_true, y_pred, labels=REACTIONS, average="macro",
                    sample_weight=sample_weight, zero_division=0)


def run_cv(name: str, make_model, df: pd.DataFrame, force=False, verbose=True, **info) -> dict:
    """CV по колонке fold. make_model() -> новая необученная модель.

    info — что записать в результат (семейство, параметры, заметка).
    """
    path = CV_DIR / f"{name}.joblib"
    if path.exists() and not force:
        result = joblib.load(path)
        if verbose:
            _print(result)
        return result

    y = df["reaction"].to_numpy()
    pred = np.empty(len(df), dtype=object)
    proba = None
    folds, turns = [], []
    for k, fit_idx, val_idx in iter_folds(df):
        model = make_model()
        start = time.perf_counter()
        model.fit(df.iloc[fit_idx].reset_index(drop=True))
        fit_seconds = time.perf_counter() - start

        valid = df.iloc[val_idx].reset_index(drop=True)
        start = time.perf_counter()
        if hasattr(model, "predict_with_proba"):
            labels, p = model.predict_with_proba(valid)
            if proba is None:
                proba = np.zeros((len(df), p.shape[1]))
            proba[val_idx] = p
        else:
            labels = model.predict(valid)
        predict_seconds = time.perf_counter() - start

        pred[val_idx] = np.asarray(labels, dtype=object)
        if hasattr(model, "turn_output_"):  # вероятности реплик: для подбора порога без переобучения
            turns.append((val_idx, *model.turn_output_))
        folds.append({
            "fold": k,
            "macro_f1": macro_f1(y[val_idx], pred[val_idx]),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "n_valid": len(val_idx),
        })

    result = {
        "name": name,
        "pred": pred.astype(str),
        "proba": proba,
        "folds": pd.DataFrame(folds),
        "turns": turns,
        "dialog_ids": df["dialog_id"].tolist(),
        **info,
    }
    CV_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    if verbose:
        _print(result)
    return result


def _print(result):
    f = result["folds"]
    print(f"{result['name']:<28} CV macro-F1 {f.macro_f1.mean():.4f} ± {f.macro_f1.std(ddof=0):.4f}"
          f"  по фолдам {np.round(f.macro_f1.to_numpy(), 3)}"
          f"  fit {f.fit_seconds.mean():.1f} с, predict {f.predict_seconds.mean():.2f} с")


def load_result(name: str) -> dict:
    return joblib.load(CV_DIR / f"{name}.joblib")


def summary_row(result: dict, y) -> dict:
    """Строка сводной таблицы: среднее по фолдам и метрики по OOF."""
    f, pred = result["folds"], result["pred"]
    return {
        "model": result["name"],
        "family": result.get("family", ""),
        "cv_macro_f1": f.macro_f1.mean(),
        "std": f.macro_f1.std(ddof=0),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "recall_positive": recall_score(y, pred, labels=["positive"], average="macro", zero_division=0),
        "recall_negative": recall_score(y, pred, labels=["negative"], average="macro", zero_division=0),
        "fit_seconds": f.fit_seconds.mean(),
        "predict_seconds": f.predict_seconds.mean(),
    }


def summary_table(results: list[dict], y) -> pd.DataFrame:
    rows = [summary_row(r, y) for r in results]
    return pd.DataFrame(rows).set_index("model").sort_values("cv_macro_f1", ascending=False)


def fold_scores(results: list[dict]) -> pd.DataFrame:
    """Таблица фолд x модель с macro-F1."""
    return pd.DataFrame({r["name"]: r["folds"].macro_f1.to_numpy() for r in results})


def paired_delta(a: dict, b: dict) -> pd.DataFrame:
    """macro-F1 модели a минус b на каждом фолде."""
    sa, sb = a["folds"].macro_f1.to_numpy(), b["folds"].macro_f1.to_numpy()
    table = pd.DataFrame({a["name"]: sa, b["name"]: sb, "delta": sa - sb})
    table.index.name = "fold"
    table.loc["mean"] = table.mean()
    return table
