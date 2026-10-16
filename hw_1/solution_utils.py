"""Функции для solution.ipynb.

текст как в задаче 18 лабы 4: буквы, стоп-слова, Snowball.
стоп-слова лежат в artifacts/, интернет не нужен.
реакция грузится из joblib в artifacts/.
"""

import csv
import json
import re
from pathlib import Path

import joblib
from nltk.stem import SnowballStemmer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

LETTERS = re.compile(r"[^\W\d_]+")
ASCII_MAX = 127

REACTION_MODEL_FILES = {
    "Naive Bayes": "naive_bayes.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "SVM": "svm.joblib",
    "decision_2x": "decision_2x.joblib",
    "score_stacking_lr": "score_stacking_lr.joblib",
}

TOPIC_MODEL_NAMES = (
    "dummy",
    "bow_nb",
    "tfidf_logreg",
    "tfidf_linearsvc",
    "tfidf_bigram_sublinear",
)


def _read_stopwords(name):
    path = ARTIFACTS / f"stopwords_{name}.txt"
    words = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            word = line.strip()
            if word:
                words.append(word)
    return set(words)


STOPWORDS = {
    "english": _read_stopwords("english"),
    "russian": _read_stopwords("russian"),
}
STEMMERS = {
    "english": SnowballStemmer("english"),
    "russian": SnowballStemmer("russian"),
}


def mostly_russian(letters):
    if not letters:
        return False
    non_ascii = 0
    for ch in letters:
        if ord(ch) > ASCII_MAX:
            non_ascii += 1
    return non_ascii > len(letters) / 2


def normalize(text):
    words = LETTERS.findall(text.lower())
    letters = "".join(words)
    language = "russian" if mostly_russian(letters) else "english"
    stop = STOPWORDS[language]
    stemmer = STEMMERS[language]
    out = []
    for word in words:
        if len(word) < 2:
            continue
        if word in stop:
            continue
        out.append(stemmer.stem(word))
    return out


def dialog_text(dialog):
    parts = []
    for turn in dialog["turns"]:
        if turn["role"] == "user":
            parts.append(turn["text"])
    return " ".join(normalize(" ".join(parts)))


def clean_text(text):
    text = re.sub(r"[\u3400-\u9fff]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def reaction_dialog_text(dialog):
    text = "\n".join(
        turn["role"] + ": " + turn["text"]
        for turn in dialog["turns"]
    )
    return clean_text(text)


def read_dialogs(path):
    dialogs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            dialogs.append({"dialog_id": record["dialog_id"], "turns": record["turns"]})
    return dialogs


def read_labels(path, task):
    labels = {}
    if not path:
        return labels
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            label = (record.get("labels") or {}).get(task)
            if label is not None:
                labels[record["dialog_id"]] = label
    return labels


def init_reaction_models(train_path):
    _ = train_path
    models = {}
    for name, filename in REACTION_MODEL_FILES.items():
        obj = joblib.load(ARTIFACTS / filename)
        # два файла - словари без predict, иначе solution.ipynb падает
        if hasattr(obj, "predict"):
            models[name] = obj
    return models


def predict_reaction(model, dialogs):
    texts = []
    for dialog in dialogs:
        texts.append(reaction_dialog_text(dialog))
    pred = model.predict(texts)
    out = []
    for label in pred:
        out.append(str(label))
    return out


def init_topic_models(train_path):
    # train_path не используем: темы уже в joblib, на проверке кластеризовать нельзя
    _ = train_path
    models = {}
    for name in TOPIC_MODEL_NAMES:
        path = ARTIFACTS / f"topic_{name}.joblib"
        models[name] = joblib.load(path)
    return models


def predict_topic(model, dialogs):
    texts = []
    for dialog in dialogs:
        texts.append(dialog_text(dialog))
    pred = model.predict(texts)
    out = []
    for label in pred:
        out.append(str(label))
    return out


def _training_data(train_path, task):
    labels = read_labels(train_path, task)
    dialogs = []
    for dialog in read_dialogs(train_path):
        if dialog["dialog_id"] in labels:
            dialogs.append(dialog)
    texts = []
    targets = []
    for dialog in dialogs:
        texts.append(dialog_text(dialog))
        targets.append(labels[dialog["dialog_id"]])
    return texts, targets


def compute_metrics(dialogs, predictions, labels):
    y_true = []
    y_pred = []
    for dialog, pred in zip(dialogs, predictions):
        if dialog["dialog_id"] in labels:
            y_true.append(labels[dialog["dialog_id"]])
            y_pred.append(pred)
    return {"accuracy": accuracy_score(y_true, y_pred)}


def _purity(y_true, y_pred):
    buckets = {}
    for true, pred in zip(y_true, y_pred):
        if pred not in buckets:
            buckets[pred] = []
        buckets[pred].append(true)
    correct = 0
    for members in buckets.values():
        best = 0
        for label in set(members):
            n = members.count(label)
            if n > best:
                best = n
        correct += best
    if not y_true:
        return 0.0
    return correct / len(y_true)


def compute_partition_metrics(dialogs, predictions, labels):
    y_true = []
    y_pred = []
    for dialog, pred in zip(dialogs, predictions):
        if dialog["dialog_id"] in labels:
            y_true.append(labels[dialog["dialog_id"]])
            y_pred.append(pred)
    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": h,
        "completeness": c,
        "v_measure": v,
        "purity": _purity(y_true, y_pred),
    }


def save_predictions(path, dialogs, predictions):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for task, by_model in predictions.items():
            for name, answers in by_model.items():
                for dialog, answer in zip(dialogs, answers):
                    writer.writerow([dialog["dialog_id"], task, name, answer])
