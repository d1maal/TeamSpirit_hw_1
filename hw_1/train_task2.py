"""пересборка тем и классификаторов.

1) тот же tf-idf + pca + k-means на 14, что в ноутбуке.
2) второй k-means только на кластере разное.
3) классификаторы из лабы 4 и семинара 3. своя: биграммы + sublinear + сетка C.
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from solution_utils import ARTIFACTS, TOPIC_MODEL_NAMES, dialog_text

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "hw1_train.jsonl"
FIG_DIR = HERE / "task2_figures"
SEED = 42
C_GRID = [0.01, 0.1, 1, 10]

FIRST_NAMES = {
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

# второй k-means k=5 на кластере 9, seed 42.
# номера проверили руками по словам и примерам.
LEFTOVER_TO_TOPIC = {
    0: "разное",
    1: "списки и примеры",
    2: "списки и примеры",
    3: "перевод",
    4: "фанфики",
}


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def make_text_model(name, C=1):
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
    if name == "tfidf_linearsvc":
        return make_pipeline(
            TfidfVectorizer(min_df=2),
            LinearSVC(max_iter=2000),
        )
    if name == "tfidf_bigram_sublinear":
        return make_pipeline(
            TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2)),
            LogisticRegression(C=C, max_iter=2000),
        )
    raise ValueError(name)


def group_ids(texts):
    seen = {}
    groups = []
    next_id = 0
    for text in texts:
        if text not in seen:
            seen[text] = next_id
            next_id += 1
        groups.append(seen[text])
    return groups


def split_by_group(texts, topics, groups, test_size=0.2):
    """один и тот же текст целиком в train или в test. как группы в лабе 14-15."""
    topic_of_group = {}
    for g, topic in zip(groups, topics):
        topic_of_group[g] = topic

    unique_groups = list(topic_of_group.keys())
    unique_topics = []
    for g in unique_groups:
        unique_topics.append(topic_of_group[g])

    counts = {}
    for t in unique_topics:
        counts[t] = counts.get(t, 0) + 1
    can_stratify = min(counts.values()) >= 2

    if can_stratify:
        g_train, g_test = train_test_split(
            unique_groups,
            test_size=test_size,
            random_state=SEED,
            stratify=unique_topics,
        )
    else:
        g_train, g_test = train_test_split(
            unique_groups,
            test_size=test_size,
            random_state=SEED,
        )

    train_set = set(g_train)
    test_set = set(g_test)
    texts_train = []
    y_train = []
    texts_test = []
    y_test = []
    for text, topic, g in zip(texts, topics, groups):
        if g in train_set:
            texts_train.append(text)
            y_train.append(topic)
        elif g in test_set:
            texts_test.append(text)
            y_test.append(topic)
    return texts_train, texts_test, y_train, y_test


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    records = load_records(DATA_PATH)
    dialogs = []
    for record in records:
        dialogs.append({"dialog_id": record["dialog_id"], "turns": record["turns"]})

    texts = []
    for dialog in dialogs:
        texts.append(dialog_text(dialog))
    print("диалогов:", len(texts))

    tfidf = TfidfVectorizer(min_df=5, max_df=0.9, sublinear_tf=True)
    matrix = tfidf.fit_transform(texts)
    dense = matrix.toarray()
    pca = PCA(n_components=350, random_state=SEED).fit(dense)
    Z = pca.transform(dense)[:, :200]

    km = KMeans(n_clusters=14, n_init=10, random_state=SEED).fit(Z)
    first = km.labels_

    leftover_idx = []
    leftover_rows = []
    for i, c in enumerate(first):
        if c == 9:
            leftover_idx.append(i)
            leftover_rows.append(Z[i])
    leftover_rows = np.array(leftover_rows)

    leftover_km = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit(leftover_rows)
    leftover_labels = leftover_km.labels_

    words = np.array(tfidf.get_feature_names_out())
    leftover_idx_arr = np.array(leftover_idx)
    print("второй k-means на кластере 9:")
    for part in range(5):
        in_part = leftover_labels == part
        mean_weight = np.asarray(matrix[leftover_idx_arr[in_part]].mean(axis=0)).ravel()
        top = words[mean_weight.argsort()[::-1][:8]]
        print(part, int(in_part.sum()), " ".join(top))

    topics = []
    for i, c in enumerate(first):
        if c != 9:
            topics.append(FIRST_NAMES[int(c)])
        else:
            topics.append(None)

    pos = 0
    for i in range(len(topics)):
        if topics[i] is None:
            topics[i] = LEFTOVER_TO_TOPIC[int(leftover_labels[pos])]
            pos += 1

    print("темы после второго k-means:")
    print(pd.Series(topics).value_counts())

    groups = group_ids(texts)
    texts_train, texts_test, y_train, y_test = split_by_group(texts, topics, groups)
    print("train:", len(texts_train), "test:", len(texts_test))

    BASELINES = ["dummy", "bow_nb", "tfidf_logreg", "tfidf_linearsvc"]
    text_models = {}
    rows = []
    for name in BASELINES:
        model = make_text_model(name)
        model.fit(texts_train, y_train)
        pred = model.predict(texts_test)
        text_models[name] = model
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        n_features = len(model[0].get_feature_names_out())
        rows.append({"accuracy": acc, "macro_f1": f1, "n_features": n_features})
        print(name, "accuracy", round(acc, 3), "macro_f1", round(f1, 3))

    text_table = pd.DataFrame(rows, index=BASELINES)

    # своя: биграммы + sublinear, C только на train. семинар 4 + задача 16.
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
    print("CV по C на train:")
    print(cv_table.round(3))
    best_C = search.best_params_["logisticregression__C"]
    print("лучший C по CV:", best_C)

    plt.figure(figsize=(6, 3.5))
    plt.plot(C_GRID, cv_table.values, "o-")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("macro-F1 на кросс-валидации")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cv_C.png", dpi=140)
    plt.close()

    pred_own = search.predict(texts_test)
    text_models["tfidf_bigram_sublinear"] = search.best_estimator_
    text_table.loc["tfidf_bigram_sublinear"] = {
        "accuracy": accuracy_score(y_test, pred_own),
        "macro_f1": f1_score(y_test, pred_own, average="macro"),
        "n_features": len(search.best_estimator_[0].get_feature_names_out()),
    }
    print(text_table.round(3))

    plt.figure(figsize=(8, 3.8))
    text_table["macro_f1"].plot.bar()
    plt.ylabel("macro-F1 на hold-out")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_macro_f1.png", dpi=140)
    plt.close()

    # C уже выбран на train. hold-out только смотрим, победителя по нему не берем.
    best_name = "tfidf_bigram_sublinear"
    if text_table.loc["tfidf_logreg", "macro_f1"] > text_table.loc[best_name, "macro_f1"]:
        print("на hold-out логистика выше своей, в отчёте это пишем")
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

    counts = pd.Series(topics).value_counts().sort_values()
    plt.figure(figsize=(8, 4.5))
    counts.plot.barh()
    plt.xlabel("число диалогов")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "topic_sizes.png", dpi=140)
    plt.close()

    best_pair = None
    best_count = -1
    for i, topic_a in enumerate(topic_order):
        for topic_b in topic_order[i + 1 :]:
            count = int(text_confusion.loc[topic_a, topic_b] + text_confusion.loc[topic_b, topic_a])
            if count > best_count:
                best_count = count
                best_pair = (topic_a, topic_b)
    print("своя модель на hold-out, чаще путает:", best_pair, best_count)

    save_names = list(TOPIC_MODEL_NAMES)
    for name in save_names:
        if name == "tfidf_bigram_sublinear":
            model = make_text_model(name, C=best_C)
        else:
            model = make_text_model(name)
        model.fit(texts, topics)
        joblib.dump(model, ARTIFACTS / f"topic_{name}.joblib")
        print("сохранено", name)

    # старый файл больше не сдаем
    old = ARTIFACTS / "topic_tfidf_sublinear_C10.joblib"
    if old.exists():
        old.unlink()

    catalog = {
        "topics": [],
        "metrics": {
            "holdout": text_table.round(4).to_dict(orient="index"),
            "own_model": "tfidf_bigram_sublinear",
            "best_C": best_C,
            "cv_table": {str(k): round(float(v), 4) for k, v in cv_table.items()},
            "most_confused_pair": list(best_pair),
            "most_confused_count": best_count,
            "n_topics": len(topic_order),
            "raznoe_n": int((pd.Series(topics) == "разное").sum()),
        },
    }
    vc = pd.Series(topics).value_counts()
    for name, n in vc.items():
        catalog["topics"].append({"name": name, "n": int(n), "share": round(int(n) / len(topics), 4)})
    (ARTIFACTS / "topic_metrics.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS / "topic_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("готово")


if __name__ == "__main__":
    main()
