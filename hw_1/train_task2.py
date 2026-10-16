"""то же обучение, что в конце task2_topics.ipynb.

нужен, только если не хочется ждать Run All ноутбука, а artifacts
надо перезаписать
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from solution_utils import ARTIFACTS, dialog_text

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "hw1_train.jsonl"
FIG_DIR = HERE / "task2_figures"
SEED = 42
K_TOPICS = 14
PCA_FIT = 350
PCA_USE = 200

# Та же сетка, что C_grid в задаче 16 лабораторной 4.
C_GRID = [0.01, 0.1, 1, 10]

TOPIC_NAMES = {
    0: "деловые тексты",
    1: "обучение моделей",
    2: "история и политика",
    3: "трейдинг",
    4: "промпты",
    5: "таблицы",
    6: "рассказы",
    7: "перевод",
    8: "код со строгими типами",
    9: "разное",
    10: "написание текстов",
    11: "задачи с числами",
    12: "программирование",
    13: "python",
}


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def user_text(dialog):
    parts = []
    for turn in dialog["turns"]:
        if turn["role"] == "user":
            parts.append(turn["text"])
    return "\n".join(parts)


def make_text_model(name):
    """Ровно те пайплайны, что в задаче 19 лабораторной 4, плюс одна своя."""
    if name == "dummy":
        return make_pipeline(
            CountVectorizer(),
            DummyClassifier(strategy="most_frequent"),
        )
    if name == "bow_nb":
        return make_pipeline(
            CountVectorizer(min_df=2),
            MultinomialNB(),
        )
    if name == "tfidf_logreg":
        return make_pipeline(
            TfidfVectorizer(min_df=2),
            LogisticRegression(max_iter=2000),
        )
    if name == "tfidf_logreg_C10":
        return make_pipeline(
            TfidfVectorizer(min_df=2),
            LogisticRegression(C=10, max_iter=2000),
        )
    if name == "tfidf_sublinear_C10":
        return make_pipeline(
            TfidfVectorizer(min_df=2, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=2000),
        )
    raise ValueError(name)


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    records = load_records(DATA_PATH)
    dialogs = []
    for record in records:
        dialogs.append({"dialog_id": record["dialog_id"], "turns": record["turns"]})

    texts = []
    dialog_ids = []
    task_type = []
    for record, dialog in zip(records, dialogs):
        texts.append(dialog_text(dialog))
        dialog_ids.append(dialog["dialog_id"])
        labels = record.get("labels") or {}
        task_type.append(labels.get("task_type"))

    print("диалогов:", len(texts))

    # --- кластеры: тот же пайплайн, что в ноутбуке ---
    tfidf = TfidfVectorizer(min_df=5, max_df=0.9, sublinear_tf=True)
    matrix = tfidf.fit_transform(texts)
    dense = matrix.toarray()
    pca = PCA(n_components=PCA_FIT, random_state=SEED).fit(dense)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    Z = pca.transform(dense)[:, :PCA_USE]

    plt.figure(figsize=(8, 3.5))
    plt.plot(np.arange(1, PCA_FIT + 1), cumulative)
    plt.axvline(PCA_USE, color="black", ls="--", lw=1)
    plt.xlabel("число компонент")
    plt.ylabel("накопленная доля дисперсии")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_variance.png", dpi=140)
    plt.close()

    # график k как на семинаре 5–6: инерция и силуэт
    k_values = list(range(6, 21, 2))
    inertias = []
    silhouettes = []
    rng = np.random.default_rng(SEED)
    sil_idx = rng.choice(len(Z), size=2000, replace=False)
    for k in k_values:
        fit = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        inertias.append(fit.inertia_)
        silhouettes.append(silhouette_score(Z[sil_idx], fit.labels_[sil_idx]))
        print("k =", k, "инерция =", round(fit.inertia_), "силуэт =", round(silhouettes[-1], 3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(k_values, inertias, "o-")
    axes[0].axvline(K_TOPICS, color="black", ls="--", lw=1)
    axes[0].set_xlabel("число кластеров k")
    axes[0].set_ylabel("инерция")
    axes[1].plot(k_values, silhouettes, "o-")
    axes[1].axvline(K_TOPICS, color="black", ls="--", lw=1)
    axes[1].set_xlabel("число кластеров k")
    axes[1].set_ylabel("силуэт")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "k_inertia_silhouette.png", dpi=140)
    plt.close()

    km = KMeans(n_clusters=K_TOPICS, n_init=10, random_state=SEED).fit(Z)
    cluster_ids = km.labels_
    topics = []
    for c in cluster_ids:
        topics.append(TOPIC_NAMES[int(c)])

    counts = pd.Series(topics).value_counts()
    plt.figure(figsize=(8, 4.5))
    counts.sort_values().plot.barh()
    plt.xlabel("число диалогов")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "topic_sizes.png", dpi=140)
    plt.close()

    ward_idx = rng.choice(len(Z), size=2000, replace=False)
    ward = AgglomerativeClustering(n_clusters=K_TOPICS, linkage="ward").fit_predict(Z[ward_idx])
    print("Ward vs k-means ARI:", round(adjusted_rand_score(cluster_ids[ward_idx], ward), 3))

    labeled_true = []
    labeled_pred = []
    for i, value in enumerate(task_type):
        if value is not None:
            labeled_true.append(value)
            labeled_pred.append(cluster_ids[i])
    print("ARI vs task_type:", round(adjusted_rand_score(labeled_true, labeled_pred), 3))

    # --- классификатор: задача 19 лабораторной 4 ---
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts,
        topics,
        test_size=0.2,
        random_state=SEED,
        stratify=topics,
    )
    print("train:", len(texts_train), "test:", len(texts_test))

    # Три базовые — dummy, bow_nb, tfidf_logreg.
    # tfidf_logreg_C10 и tfidf_sublinear_C10 — те же сравнения, что в лабе.
    BASELINES = ["dummy", "bow_nb", "tfidf_logreg"]
    LAB19_EXTRA = ["tfidf_logreg_C10", "tfidf_sublinear_C10"]
    TEXT_MODELS = BASELINES + LAB19_EXTRA

    text_models = {}
    rows = []
    for name in TEXT_MODELS:
        model = make_text_model(name)
        model.fit(texts_train, y_train)
        pred = model.predict(texts_test)
        text_models[name] = model
        vectorizer = model[0]
        rows.append({
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "n_features": len(vectorizer.get_feature_names_out()),
        })
        print(name, "accuracy", round(rows[-1]["accuracy"], 3), "macro_f1", round(rows[-1]["macro_f1"], 3))

    text_table = pd.DataFrame(rows, index=TEXT_MODELS)

    # Своя модель: sublinear + биграммы, C подбираем GridSearchCV как в задаче 16.
    # Метрика поиска — macro-F1, потому что тем несколько и они неравные
    # (в лабе 16 был roc_auc, там задача была бинарная).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        make_pipeline(
            TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000),
        ),
        param_grid={"logisticregression__C": C_GRID},
        cv=cv,
        scoring="f1_macro",
    )
    search.fit(texts_train, y_train)

    cv_table = pd.Series(dtype=float)
    for C, mean_score in zip(C_GRID, search.cv_results_["mean_test_score"]):
        cv_table.loc[C] = mean_score
    print("CV по C:")
    print(cv_table.round(3))
    print("лучший C:", search.best_params_["logisticregression__C"])

    plt.figure(figsize=(6, 3.5))
    plt.plot(C_GRID, cv_table.values, "o-")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("macro-F1 на кросс-валидации")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cv_C.png", dpi=140)
    plt.close()

    pred_own = search.predict(texts_test)
    text_models["tfidf_bigram_Csearch"] = search.best_estimator_
    text_table.loc["tfidf_bigram_Csearch"] = {
        "accuracy": accuracy_score(y_test, pred_own),
        "macro_f1": f1_score(y_test, pred_own, average="macro"),
        "n_features": len(search.best_estimator_[0].get_feature_names_out()),
    }
    print("\nтаблица")
    print(text_table.round(3))

    plt.figure(figsize=(8, 3.8))
    text_table["macro_f1"].plot.bar()
    plt.ylabel("macro-F1 на hold-out")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_macro_f1.png", dpi=140)
    plt.close()

    best_name = text_table["macro_f1"].idxmax()
    best_pred = text_models[best_name].predict(texts_test)
    topic_order = sorted(set(topics))
    text_confusion = pd.DataFrame(
        confusion_matrix(y_test, best_pred, labels=topic_order),
        index=topic_order,
        columns=topic_order,
    )

    plt.figure(figsize=(9, 8))
    plt.imshow(text_confusion.values, cmap="Blues")
    plt.xticks(range(len(topic_order)), topic_order, rotation=90)
    plt.yticks(range(len(topic_order)), topic_order)
    plt.xlabel("предсказанная тема")
    plt.ylabel("настоящая тема")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion.png", dpi=140)
    plt.close()

    best_pair = None
    best_count = -1
    for i, topic_a in enumerate(topic_order):
        for topic_b in topic_order[i + 1 :]:
            count = int(text_confusion.loc[topic_a, topic_b] + text_confusion.loc[topic_b, topic_a])
            if count > best_count:
                best_count = count
                best_pair = (topic_a, topic_b)
    print("лучшая модель:", best_name)
    print("чаще всего путает:", best_pair, "ошибок", best_count)

    # Для сдачи переучиваем выбранные модели на всех 10k.
    # В artifacts кладём три базовые и свою.
    save_names = ["dummy", "bow_nb", "tfidf_logreg", "tfidf_sublinear_C10"]
    best_C = search.best_params_["logisticregression__C"]
    for name in save_names:
        model = make_text_model(name)
        model.fit(texts, topics)
        joblib.dump(model, ARTIFACTS / f"topic_{name}.joblib")
        print("сохранено", name)

    catalog = {
        "metrics": {
            "holdout": text_table.round(4).to_dict(orient="index"),
            "best_model": best_name,
            "best_C": best_C,
            "cv_table": {str(k): round(float(v), 4) for k, v in cv_table.items()},
            "most_confused_pair": list(best_pair),
            "most_confused_count": best_count,
        }
    }
    (ARTIFACTS / "topic_metrics.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
