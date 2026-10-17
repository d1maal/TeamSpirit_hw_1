"""Функции, которые импортирует solution.ipynb.
брал как из четвертой лабы. остальное добавил сам.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.dummy import DummyClassifier
from reaction_models import Decision2xClassifier, ScoreStackingLRClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

LETTERS = re.compile(r"[^\W\d_]+")
ASCII_MAX = 127
STOPWORDS = {
    language: set(words)
    for language, words in json.loads((ARTIFACTS / "nltk_stopwords.json").read_text(encoding="utf-8")).items()
}

STEMMERS = {
    "english": SnowballStemmer("english"),
    "russian": SnowballStemmer("russian"),
}

TOPIC_MODEL_NAMES = (
    "dummy",
    "bow_nb",
    "tfidf_logreg",
    "tfidf_linearsvc",
    "tfidf_bigram_sublinear",
)


def mostly_russian(letters: str) -> bool:
    """Больше половины букв вне ASCII - ставим русский стеммер"""
    if not letters:
        return False
    non_ascii = sum(1 for ch in letters if ord(ch) > ASCII_MAX)
    return non_ascii > len(letters) / 2


def normalize(text: str) -> list[str]:
    """нижний регистр, слова из букв, без стоп-слов, основа слов"""
    words = LETTERS.findall(text.lower())
    letters = "".join(words)
    language = "russian" if mostly_russian(letters) else "english"
    stop, stemmer = STOPWORDS[language], STEMMERS[language]
    return [stemmer.stem(word) for word in words if len(word) >= 2 and word not in stop]


def dialog_text(dialog: dict) -> str:
    """реплики одной строкой из нормализованных слов."""
    user_parts = [turn["text"] for turn in dialog["turns"] if turn["role"] == "user"]
    return " ".join(normalize(" ".join(user_parts)))

def clean_text(text):
    text = re.sub(r"[\u3400-\u9fff]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def reaction_dialog_text(dialog: dict) -> str:
    text = "\n".join(
        turn["role"] + ": " + turn["text"]
        for turn in dialog["turns"]
    )
    return clean_text(text)


def read_dialogs(path: str) -> list[dict]:
    dialogs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                dialogs.append({"dialog_id": record["dialog_id"], "turns": record["turns"]})
    return dialogs


def read_labels(path: str, task: str) -> dict[str, str]:
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            label = (record.get("labels") or record).get(task)
            if label is not None:
                labels[record["dialog_id"]] = label
    return labels



def init_reaction_models(train_path: str) -> dict:
    """Загружаем обученные модели первой задачи."""
    del train_path

    filenames = {"Naive Bayes": "naive_bayes.joblib", "Logistic Regression": "logistic_regression.joblib", "SVM": "svm.joblib", 
                 "decision_2x": "decision_2x.joblib", "score_stacking_lr": "score_stacking_lr.joblib"}
    models = {}

    base = joblib.load(ARTIFACTS / filenames["decision_2x"])
    stack = joblib.load(ARTIFACTS / filenames["score_stacking_lr"])
    for name, filename in filenames.items():
        if name == "decision_2x":
            models[name] = Decision2xClassifier(base)
        elif name == "score_stacking_lr":
            models[name] = ScoreStackingLRClassifier(base, stack)
        else:
            models[name] = joblib.load(ARTIFACTS / filename)

    return models



def predict_reaction(model, dialogs: list[dict]) -> list[str]:
    if not dialogs:
        return []
    if isinstance(model, (Decision2xClassifier, ScoreStackingLRClassifier)):
        return model.predict(dialogs)
    texts = [reaction_dialog_text(dialog) for dialog in dialogs]
    return [str(label) for label in model.predict(texts)]


def init_topic_models(train_path: str) -> dict:
    """классификаторы тем из artifacts"""
    del train_path
    models = {}
    for name in TOPIC_MODEL_NAMES:
        path = ARTIFACTS / f"topic_{name}.joblib"
        models[name] = joblib.load(path)
    return models


def predict_topic(model, dialogs: list[dict]) -> list[str]:
    texts = [dialog_text(dialog) for dialog in dialogs]
    return [str(label) for label in model.predict(texts)]


def _training_data(train_path: str, task: str) -> tuple[list[str], list[str]]:
    labels = read_labels(train_path, task)
    dialogs = [dialog for dialog in read_dialogs(train_path) if dialog["dialog_id"] in labels]
    texts = [dialog_text(dialog) for dialog in dialogs]
    targets = [labels[dialog["dialog_id"]] for dialog in dialogs]
    return texts, targets


def compute_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    y_true = [true for true, _ in pairs]
    y_pred = [pred for _, pred in pairs]
    if not pairs:
        return {}
    report = classification_report(
        y_true, y_pred, labels=["negative", "neutral", "positive"],
        output_dict=True, zero_division=0,
    )
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, labels=["negative", "neutral", "positive"], average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    for name in ["negative", "neutral", "positive"]:
        for metric in ["precision", "recall", "f1-score"]:
            metrics[f"{name}_{metric}"] = report[name][metric]
    return metrics


def _purity(y_true: list[str], y_pred: list[str]) -> float:
    """Доля объектов, у которых в их предсказанной теме самая частая истинная метка"""
    buckets: dict[str, list[str]] = {}
    for true, pred in zip(y_true, y_pred):
        buckets.setdefault(pred, []).append(true)
    correct = 0
    for members in buckets.values():
        correct += max(members.count(label) for label in set(members))
    return correct / len(y_true) if y_true else 0.0


def compute_partition_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    """ считать ARI, NMI, однородность, полнота, V-мера, чистота."""
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    y_true = [true for true, _ in pairs]
    y_pred = [pred for _, pred in pairs]
    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
    return {"ARI": adjusted_rand_score(y_true, y_pred), "NMI": normalized_mutual_info_score(y_true, y_pred), "homogeneity": h, "completeness": c, "v_measure": v, "purity": _purity(y_true, y_pred)}


def save_predictions(path: str, dialogs: list[dict], predictions: dict[str, dict[str, list[str]]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for task, by_model in predictions.items():
            for name, answers in by_model.items():
                for dialog, answer in zip(dialogs, answers):
                    writer.writerow([dialog["dialog_id"], task, name, answer])
