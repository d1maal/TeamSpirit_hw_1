"""Функции, которые импортирует solution.ipynb (HW2, только реакция).

Модели делятся по префиксу: hw1_ — модели задачи 1 HW1 без изменений,
hw2_ — новые, на деревьях и ансамблях. Все грузятся из artifacts/.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

from reaction_models import Decision2xClassifier, ScoreStackingLRClassifier

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
REACTIONS = ["negative", "neutral", "positive"]

# пайплайны sklearn на тексте диалога (TF-IDF + линейная модель)
HW1_TEXT_MODELS = {
    "hw1_naive_bayes": "hw1_naive_bayes.joblib",
    "hw1_logistic_regression": "hw1_logistic_regression.joblib",
    "hw1_svm": "hw1_svm.joblib",
}
# модели по репликам: (базовая часть, стекинг или None)
HW1_DIALOG_MODELS = {
    "hw1_decision_2x": ("hw1_decision_2x.joblib", None),
    "hw1_score_stacking_lr": ("hw1_decision_2x.joblib", "hw1_score_stacking_lr.joblib"),
    "hw1_grisha_909": ("hw1_grisha_909.joblib", None),
    "hw1_stack_909": ("hw1_grisha_909.joblib", "hw1_stack_909.joblib"),
}
# новые модели (деревья, бустинг, стекинг): имя -> файл
HW2_MODELS = {name: f"{name}.joblib" for name in ["hw2_random_forest", "hw2_hgb", "hw2_lightgbm", "hw2_stack", "hw2_stack_prior"]}


# ---------------------------------------------------------------------------
#  Данные
# ---------------------------------------------------------------------------

def read_dialogs(path: str) -> list[dict]:
    dialogs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                dialogs.append({"dialog_id": record["dialog_id"], "turns": record["turns"]})
    return dialogs


def read_labels(path: str) -> dict[str, str]:
    """dialog_id -> реакция. Понимает и обучающий формат (labels.reaction),
    и плоский формат hidden_labels.jsonl (reaction на верхнем уровне)."""
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            label = (record.get("labels") or record).get("reaction")
            if label is not None:
                labels[record["dialog_id"]] = label
    return labels


# ---------------------------------------------------------------------------
#  Обработка текста для моделей HW1 на всем диалоге
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"[㐀-鿿]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def reaction_dialog_text(dialog: dict) -> str:
    text = "\n".join(turn["role"] + ": " + turn["text"] for turn in dialog["turns"])
    return clean_text(text)


# ---------------------------------------------------------------------------
#  Модели
# ---------------------------------------------------------------------------

def init_reaction_models(train_path: str) -> dict:
    """Все модели из artifacts: hw1_* и hw2_*. train_path не используется."""
    del train_path
    models = {}
    for name, filename in HW1_TEXT_MODELS.items():
        models[name] = joblib.load(ARTIFACTS / filename)
    loaded = {}
    for name, (base_file, stack_file) in HW1_DIALOG_MODELS.items():
        if base_file not in loaded:
            loaded[base_file] = joblib.load(ARTIFACTS / base_file)
        base = loaded[base_file]
        if stack_file is None:
            models[name] = Decision2xClassifier(base)
        else:
            models[name] = ScoreStackingLRClassifier(base, joblib.load(ARTIFACTS / stack_file))
    for name, filename in HW2_MODELS.items():
        models[name] = joblib.load(ARTIFACTS / filename)
    return models


def predict_reaction(model, dialogs: list[dict]) -> list[str]:
    """По одному ответу на диалог: positive, neutral или negative."""
    if not dialogs:
        return []
    if isinstance(model, (Decision2xClassifier, ScoreStackingLRClassifier)):
        return [str(p) for p in model.predict(dialogs)]
    if hasattr(model, "predict_dialogs"):  # новые модели на признаках диалога
        return [str(p) for p in model.predict_dialogs(dialogs)]
    texts = [reaction_dialog_text(dialog) for dialog in dialogs]
    return [str(label) for label in model.predict(texts)]


# ---------------------------------------------------------------------------
#  Метрики и сохранение
# ---------------------------------------------------------------------------

def compute_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    if not pairs:
        return {}
    y_true = [t for t, _ in pairs]
    y_pred = [p for _, p in pairs]
    report = classification_report(y_true, y_pred, labels=REACTIONS, output_dict=True, zero_division=0)
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, labels=REACTIONS, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    for name in REACTIONS:
        metrics[f"{name}_precision"] = report[name]["precision"]
        metrics[f"{name}_recall"] = report[name]["recall"]
    return metrics


def save_predictions(path: str, dialogs: list[dict], predictions: dict[str, list[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for name, answers in predictions.items():
            for dialog, answer in zip(dialogs, answers):
                writer.writerow([dialog["dialog_id"], "reaction", name, answer])
