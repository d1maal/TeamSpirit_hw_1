"""Все эксперименты HW2: имя -> (семейство, фабрика модели).

Запуск в фоне: python experiments.py — считает все модели на 5 фолдах
и кэширует результаты в artifacts/cv/. Ноутбук читает кэш.
"""

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import old_models as om
from turn_stack import TurnStack
from trees import BaggedDecision2x, DialogTextModel, OOFStack, TurnModel


def forest(n=100, leaf=3, max_features="sqrt"):
    return RandomForestClassifier(n, min_samples_leaf=leaf, max_features=max_features, n_jobs=-1, random_state=42)


def lgbm(depth=4, lr=0.07, n=200):
    return LGBMClassifier(max_depth=depth, num_leaves=2 ** depth, learning_rate=lr, n_estimators=n,
                          verbosity=-1, random_state=42)


def catboost():
    return CatBoostClassifier(depth=4, learning_rate=0.07, iterations=200, verbose=False,
                              allow_writing_files=False, random_seed=42)


# 4. Деревья с параметрами из ноутбука коллег
TREES = {
    "Tree": lambda: TurnModel(DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)),
    "Bagging": lambda: TurnModel(BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, n_jobs=-1, random_state=42)),
    "Forest": lambda: TurnModel(forest()),
    "Forest_tfidf": lambda: TurnModel(forest(), kind="raw"),
    "HGB": lambda: TurnModel(HistGradientBoostingClassifier(max_depth=4, learning_rate=0.07, max_iter=200,
                                                            early_stopping=False, random_state=42)),
    "LightGBM": lambda: TurnModel(lgbm()),
    "CatBoost": lambda: TurnModel(catboost()),
}

# 5. Подбор: варианты лучших семейств и заведомо неудачные попытки
TUNING = {
    "Forest_tfidf_leaf1": lambda: TurnModel(forest(leaf=1), kind="raw"),
    "Forest_tfidf_leaf6": lambda: TurnModel(forest(leaf=6), kind="raw"),
    "Forest_tfidf_mf0.02": lambda: TurnModel(forest(max_features=0.02), kind="raw"),
    "Forest_tfidf_300": lambda: TurnModel(forest(n=300), kind="raw"),
    "Forest_tfidf_unbalanced": lambda: TurnModel(forest(), kind="raw", balanced=False),
    "LightGBM_d3": lambda: TurnModel(lgbm(depth=3)),
    "LightGBM_d6": lambda: TurnModel(lgbm(depth=6)),
    "LightGBM_lr0.03_n600": lambda: TurnModel(lgbm(lr=0.03, n=600)),
    "LightGBM_tfidf": lambda: TurnModel(lgbm(), kind="raw"),
    "LightGBM_svm": lambda: TurnModel(lgbm(), svm=True),
    "Forest_svm": lambda: TurnModel(forest(), svm=True),
    "Forest_dialog_tfidf": lambda: DialogTextModel(forest()),
    "Bagging_decision_2x": lambda: BaggedDecision2x(),
}

# 6. Стекинг на OOF (мета — LR)
STACK_BASES = {
    "hw1_decision_2x": om.Decision2x,
    "hw1_logistic_regression": lambda: om.TextModel("hw1_logistic_regression"),
    "Forest_tfidf": TREES["Forest_tfidf"],
    "LightGBM": TREES["LightGBM"],
    "CatBoost": TREES["CatBoost"],
}
STACKS = {
    "Stack_lr": lambda: OOFStack(STACK_BASES),
    "Stack_lr_unbalanced": lambda: OOFStack(STACK_BASES, balanced=False),
    "Stack_d2x_forest": lambda: OOFStack({k: STACK_BASES[k] for k in ["hw1_decision_2x", "Forest_tfidf"]}),
}

# Лучшая новая модель: стекинг на уровне реплик
STACK_TREES = {"Forest_tfidf_300": TUNING["Forest_tfidf_300"], "LightGBM_d6": TUNING["LightGBM_d6"]}
STACKS["hw2_stack"] = lambda: TurnStack(STACK_TREES, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))

# Поправка на доли классов: (1/3) / доля класса в обучении (negative 923, neutral 1774, positive 303 из 3000)
PRIOR_WEIGHTS = [1000 / n for n in (923, 1774, 303)]

# Финальные модели hw2_* (обучаются на всех 3000)
FINAL = {
    "hw2_random_forest": TUNING["Forest_tfidf_300"],
    "hw2_hgb": TREES["HGB"],
    "hw2_lightgbm": TUNING["LightGBM_d6"],
    "hw2_stack": STACKS["hw2_stack"],
    "hw2_stack_prior": lambda: TurnStack(STACK_TREES, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                         weights=PRIOR_WEIGHTS),
}

FAMILY = {**{k: "дерево/ансамбль" for k in TREES}, **{k: "подбор" for k in TUNING},
          **{k: "стекинг" for k in STACKS}}
ALL = {**TREES, **TUNING, **STACKS}


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    from cv import run_cv
    from splits import load_data

    df = load_data()
    names = sys.argv[1:] or list(ALL)
    for name in names:
        run_cv(name, ALL[name], df, family=FAMILY[name])

