"""Собирает experiments.ipynb. Запуск: python build_notebook.py"""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# HW2. Деревья и ансамбли: реакция пользователя

Все модели оцениваются на одних и тех же 5 фолдах: `StratifiedGroupKFold(5, shuffle=True,
random_state=42)` по группам почти-дубликатов на всех 3000 размеченных диалогах.
Метрика — macro-F1 по диалогам, среднее по фолдам. Результаты CV кэшируются в
`artifacts/cv/` (`python experiments.py` считает всё заново).

Код: `splits.py` — данные и фолды, `cv.py` — цикл CV, `old_models.py` — модели HW1,
`trees.py` — признаки реплик и деревья, `turn_stack.py` — стекинг по репликам,
`experiments.py` — список экспериментов, `report.py` — графики.
""")

code("""
import json, pickle, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight

import experiments as E
import old_models as om
import trees as T
import turn_stack as S
from cv import run_cv, summary_table, paired_delta, macro_f1, REACTIONS
from report import compare_plot, paired_plot
from splits import load_data, iter_folds

pd.set_option("display.precision", 4)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
""")

# ---------------------------------------------------------------------------
md("""
## 1. Данные и фолды

Реакция диалога — последняя ненейтральная реакция реплик пользователя, иначе neutral.
Почти-дубликаты (TF-IDF реплик пользователя, косинус ≥ 0.95) объединены в группы
и не попадают в разные фолды.
""")
code("""
df = load_data()
y = df["reaction"].to_numpy()
print("диалогов:", len(df), " групп:", df["group"].nunique())
display(pd.crosstab(df["fold"], df["reaction"], margins=True))
""")

# ---------------------------------------------------------------------------
md("""
## 2. Старые модели на тех же фолдах

Модели HW1 переобучаются на каждом фолде с гиперпараметрами HW1 без изменений
(параметры из `artifacts/hw1_*.joblib` через `clone`, предсказание — тем же кодом
`reaction_models`, что в `solution.ipynb`).

Проверка воспроизведения (обучение на тех же данных, что артефакт): NB, LR, SVM
совпадают с сохранёнными на 100% предсказаний; база 909 на сохранённых
`train_dialog_ids` — 99.8%; стек 909 — 96% (код обучения 909 не сохранился,
воспроизведение приближённое). `hw1_decision_2x` и `hw1_score_stacking_lr` в
`artifacts/` обучены на 2400 диалогах train-части HW1, тот сплит не восстановить.
""")
code("""
old = {name: run_cv(name, make, df, family="HW1") for name, make in om.OLD_MODELS.items()}
""")

# ---------------------------------------------------------------------------
md("""
## 3. Признаки реплик

Объект — реплика пользователя j ≥ 1 (у реплики есть метка реакции). Признаки учатся
только на обучающей части фолда:
- `raw` — TF-IDF 42 000: user слова 1–2 (12k), user символы 3–5 (18k),
  ответ ассистента до реплики (6k, вес 0.4), ответ после (6k, вес 0.4);
- `turn` — 137 плотных: SVD(100) от `raw` + 37 числовых (структура, длины ответов,
  9 регулярок × 3 источника, пересечения слов).

Метка диалога: последняя реплика с max(P(neg), P(pos)) ≥ 0.5, класс — больший из
P(neg), P(pos); если такой нет — neutral.
""")
code("""
fit0, val0 = df[df.fold != 0].reset_index(drop=True), df[df.fold == 0].reset_index(drop=True)
start = time.perf_counter()
rep0 = T.features(fit0)
print(f"построение признаков на фолде 0: {time.perf_counter() - start:.1f} с")
print("raw:", rep0.train["raw"].shape, " turn:", rep0.train["turn"].shape)
display(rep0.train["pairs"]["reaction"].value_counts().to_frame("реплик"))
""")

# ---------------------------------------------------------------------------
md("""
## 4. Деревья и ансамбли

Параметры из ноутбука коллег, у всех веса классов `balanced`: Tree (depth 6, leaf 10),
Bagging (100 полных деревьев), Forest (100, leaf 3, sqrt), Forest_tfidf (то же на `raw`),
HGB / LightGBM / CatBoost (depth 4, lr 0.07, 200 итераций).
""")
code("""
new = {name: run_cv(name, make, df, family=E.FAMILY[name]) for name, make in E.TREES.items()}
display(summary_table(list(new.values()), y).drop(columns=["fit_seconds", "predict_seconds"]))
""")

# ---------------------------------------------------------------------------
md("""
## 5. Путь подбора, включая неудачи

Каждая строка — одно изменение относительно «родителя». Все варианты оценены на тех же
5 фолдах. Выбор по внешней CV даёт оптимистичную оценку выбранного варианта: при
разбросе фолдов ±0.01–0.02 разница меньше ~0.01 не значима.
""")
code("""
new.update({name: run_cv(name, make, df, family=E.FAMILY[name], verbose=False) for name, make in E.ALL.items()})
res = {**old, **new}
f1 = lambda n: res[n]["folds"].macro_f1.to_numpy()

PATH = [
    ("Forest_tfidf", "Forest", "TF-IDF вместо SVD-признаков"),
    ("Forest_tfidf_leaf1", "Forest_tfidf", "min_samples_leaf 3 → 1"),
    ("Forest_tfidf_leaf6", "Forest_tfidf", "min_samples_leaf 3 → 6"),
    ("Forest_tfidf_mf0.02", "Forest_tfidf", "max_features sqrt → 0.02"),
    ("Forest_tfidf_300", "Forest_tfidf", "100 → 300 деревьев"),
    ("Forest_tfidf_unbalanced", "Forest_tfidf", "без весов классов"),
    ("LightGBM_d3", "LightGBM", "глубина 4 → 3"),
    ("LightGBM_d6", "LightGBM", "глубина 4 → 6"),
    ("LightGBM_lr0.03_n600", "LightGBM", "lr 0.07 → 0.03, 600 итераций"),
    ("LightGBM_tfidf", "LightGBM", "TF-IDF вместо SVD-признаков"),
    ("LightGBM_svm", "LightGBM", "+ OOF-оценки SVM из HW1"),
    ("Forest_svm", "Forest", "+ OOF-оценки SVM из HW1"),
    ("Forest_dialog_tfidf", "Forest_tfidf", "TF-IDF всего диалога вместо реплик"),
    ("Bagging_decision_2x", "hw1_decision_2x", "бэггинг: 10 копий на бутстрепе"),
    ("Stack_lr", "hw1_decision_2x", "стек по диалогу: 5 баз, мета LR balanced"),
    ("Stack_lr_unbalanced", "Stack_lr", "мета LR без весов классов"),
    ("Stack_d2x_forest", "Stack_lr", "только Decision_2x + Forest_tfidf"),
    ("hw2_stack", "Stack_lr_unbalanced", "стек по репликам: SVM + Forest_300 + LightGBM_d6"),
]
path = pd.DataFrame([dict(model=n, parent=p, change=c, cv_macro_f1=f1(n).mean(),
                          delta_parent=f1(n).mean() - f1(p).mean(), better_folds=f"{(f1(n) > f1(p)).sum()}/5",
                          delta_decision_2x=f1(n).mean() - f1("hw1_decision_2x").mean())
                     for n, p, c in PATH]).set_index("model")
display(path)
""")

md("""
**Порог правила последней реакции** — по сохранённым вероятностям реплик, без переобучения.
""")
code("""
grid = np.arange(0.3, 0.71, 0.05)
fig, ax = plt.subplots(figsize=(6, 3.5))
for name in ["Forest_tfidf_300", "LightGBM_d6", "Forest_svm", "Forest"]:
    curve = [np.mean([macro_f1(y[idx], T.last_reaction(pos, p, len(idx), t)[0]) for idx, pos, p in res[name]["turns"]])
             for t in grid]
    ax.plot(grid, curve, marker="o", label=name)
ax.axhline(f1("hw1_decision_2x").mean(), color="#d62728", ls="--", label="hw1_decision_2x")
ax.set_xlabel("порог"); ax.set_ylabel("CV macro-F1"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "threshold.png", dpi=130); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 6. Стекинг на OOF по репликам (`hw2_stack`)

Разбор ошибок (раздел 8) показал: Decision_2x и деревья расходятся на ~13% диалогов,
и расхождение — в том, *какая реплика* считается реакцией. Поэтому стекинг строится на
уровне реплик, а не по сводкам диалога:

1. базы — два LinearSVC из `hw1_decision_2x` (старая модель), Forest_tfidf_300, LightGBM_d6;
2. вход меты для реплики — 2 оценки SVM + 3 + 3 вероятности деревьев, для обучения
   меты — out-of-fold (3 внутренних фолда по группам внутри обучающей части);
3. мета — логистическая регрессия; метка диалога — правило последней реакции с
   порогом, подобранным по OOF-вероятностям меты на обучающей части.

Варианты меты ниже считаются по один раз сохранённым выходам баз (`level1_turns`).
""")
code("""
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

L1T = S.level1_turns(df, E.STACK_TREES)
LR = lambda **kw: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **kw))
SVM, FOREST, LGBM = [0, 1], [2, 3, 4], [5, 6, 7]
variants = {
    "LR: только SVM": (LR(), SVM),
    "LR: SVM + Forest": (LR(), SVM + FOREST),
    "LR: SVM + LightGBM": (LR(), SVM + LGBM),
    "LR: SVM + Forest + LightGBM (hw2_stack)": (LR(), SVM + FOREST + LGBM),
    "LR balanced: SVM + Forest + LightGBM": (LR(class_weight="balanced"), SVM + FOREST + LGBM),
    "LightGBM-мета: SVM + Forest + LightGBM": (LGBMClassifier(max_depth=3, num_leaves=8, learning_rate=0.05,
                                                              n_estimators=200, verbosity=-1, random_state=42),
                                               SVM + FOREST + LGBM),
}
rows = []
for name, (meta, cols) in variants.items():
    pred, thr = S.oof_predict(df, L1T, meta, cols)
    s = np.array([macro_f1(y[v], pred[v]) for _, _, v in iter_folds(df)])
    rows.append(dict(meta=name, cv_macro_f1=s.mean(), std=s.std(), better_than_d2x=f"{(s > f1('hw1_decision_2x')).sum()}/5",
                     thresholds=np.round(thr, 3)))
display(pd.DataFrame(rows).set_index("meta"))
""")
md("""
Мета на одних SVM-оценках — это перекалиброванный Decision_2x; разница между ней и
полным стеком показывает вклад деревьев. Мета LightGBM в среднем по 4 seed даёт 0.731,
не лучше LR, поэтому в `hw2_stack` взята более простая LR.
""")
code("""
paired = paired_delta(res["hw2_stack"], res["hw1_decision_2x"])
display(paired)
paired_plot(paired, "hw2_stack", "hw1_decision_2x", FIG / "paired_delta.png")
""")

# ---------------------------------------------------------------------------
md("""
## 7. Сводная таблица и сравнение

Качество — по 5 фолдам. `cv_macro_f1_equal` — macro-F1 при равных долях классов
(веса объектов `balanced`), см. раздел 10.
""")
code("""
w_eq = compute_sample_weight("balanced", y)
table = summary_table(list(res.values()), y).drop(columns=["fit_seconds", "predict_seconds"])
table.insert(2, "cv_macro_f1_equal", [np.mean([macro_f1(y[v], res[n]["pred"][v], w_eq[v]) for _, _, v in iter_folds(df)])
                                      for n in table.index])
display(table)
table.to_csv("artifacts/summary_table.csv")
""")
md("""
**Время и размер** — одинаково для всех моделей на фолде 0: обучение на 2400 диалогах
с нуля (вместе с построением признаков), предсказание 600 диалогов, размер — pickle
обученной модели. Считается один раз и кэшируется.
""")
code("""
CHART = [*om.OLD_MODELS, "Tree", "Bagging", "Forest", "Forest_tfidf_300", "HGB", "LightGBM_d6", "CatBoost",
         "Stack_lr", "hw2_stack"]
FACTORY = {**om.OLD_MODELS, **E.ALL}
cost_path = Path("artifacts/fold0_cost.json")
cost = json.loads(cost_path.read_text()) if cost_path.exists() else {}
for name in CHART:
    if name not in cost:
        T._cache.clear()
        start = time.perf_counter(); model = FACTORY[name]().fit(fit0); fit_s = time.perf_counter() - start
        start = time.perf_counter(); model.predict(val0); pred_s = time.perf_counter() - start
        cost[name] = dict(fit_seconds=fit_s, predict_seconds=pred_s, size_mb=len(pickle.dumps(model)) / 2**20)
        cost_path.write_text(json.dumps(cost, indent=1))
chart = table.loc[CHART, ["family", "cv_macro_f1", "std", "cv_macro_f1_equal"]].join(pd.DataFrame(cost).T)
display(chart.sort_values("cv_macro_f1", ascending=False))
compare_plot(chart, {n: f1(n) for n in CHART}, "hw1_decision_2x", "hw2_stack", FIG / "comparison.png")
""")

# ---------------------------------------------------------------------------
md("""
## 8. Ошибки

Матрицы ошибок по OOF-предсказаниям всех 3000 диалогов (строки — истина).
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, name in zip(axes, ["hw1_decision_2x", "Forest_tfidf_300", "hw2_stack"]):
    ConfusionMatrixDisplay.from_predictions(y, res[name]["pred"], labels=REACTIONS, ax=ax, colorbar=False)
    ax.set_title(f"{name}  (macro-F1 {f1(name).mean():.3f})")
fig.tight_layout(); fig.savefig(FIG / "confusion.png", dpi=130); plt.show()

def error_types(p):
    return dict(missed=((y != "neutral") & (p == "neutral")).sum(), false_alarm=((y == "neutral") & (p != "neutral")).sum(),
                polarity=((y != "neutral") & (p != "neutral") & (y != p)).sum(), total=(y != p).sum())
display(pd.DataFrame({n: error_types(res[n]["pred"]) for n in ["hw1_decision_2x", "Forest_tfidf_300", "LightGBM_d6", "hw2_stack"]}).T)

d, t = res["hw1_decision_2x"]["pred"], res["Forest_tfidf_300"]["pred"]
diff = d != t
oracle = np.where(d == y, d, t)
print(f"Decision_2x и Forest_tfidf_300 расходятся на {diff.sum()} диалогах ({diff.mean():.1%}): "
      f"прав Decision_2x — {((d == y) & diff).sum()}, прав Forest — {((t == y) & diff).sum()}")
print(f"оракул (на каждом диалоге правильная из двух): macro-F1 {np.mean([macro_f1(y[v], oracle[v]) for _, _, v in iter_folds(df)]):.4f}")
""")

# ---------------------------------------------------------------------------
md("""
## 9. Важность признаков

На фолде 0: модели обучены на 2400 диалогах, перестановочная важность — на репликах
валидации (600 диалогов), метрика macro-F1 по репликам, 5 перестановок.
""")
code("""
forest_tfidf = E.TUNING["Forest_tfidf_300"]().fit(fit0)
mdi_words = pd.Series(forest_tfidf.model.feature_importances_, forest_tfidf.rep.names("raw")).nlargest(15)

forest = E.TREES["Forest"]().fit(fit0)
val_pairs = T.turn_pairs(val0, training=True)
x_val = forest.rep.transform(val0)["turn"]
perm = permutation_importance(forest.model, x_val, val_pairs["reaction"], scoring="f1_macro",
                              n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.DataFrame({"MDI": forest.model.feature_importances_, "permutation": perm.importances_mean},
                   index=forest.rep.names("turn"))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
mdi_words[::-1].plot.barh(ax=axes[0], color="#1f77b4", title="Forest_tfidf_300: MDI, топ-15 n-грамм")
imp["MDI"].nlargest(15)[::-1].plot.barh(ax=axes[1], color="#1f77b4", title="Forest (turn): MDI, топ-15")
imp["permutation"].nlargest(15)[::-1].plot.barh(ax=axes[2], color="#ff7f0e", title="Forest (turn): перестановочная, топ-15")
fig.tight_layout(); fig.savefig(FIG / "importance.png", dpi=130); plt.show()

imp["rank_MDI"] = imp["MDI"].rank(ascending=False).astype(int)
imp["rank_perm"] = imp["permutation"].rank(ascending=False).astype(int)
display(imp.sort_values("permutation", ascending=False).head(15))
print("SVD-компонент среди топ-15 MDI:", imp["MDI"].nlargest(15).index.str.startswith("svd").sum(),
      "| среди топ-15 перестановочной:", imp["permutation"].nlargest(15).index.str.startswith("svd").sum())
""")

# ---------------------------------------------------------------------------
md("""
## 10. Гипотеза: доли классов на скрытом тесте

В обучении классы 59/31/10% (neutral/negative/positive), на скрытом тесте по README HW1 —
1:1:1. При сдвиге долей macro-F1 моделей, которые «экономят» на редком positive, падает.
Гипотеза: (1) по macro-F1 при равных долях модели с балансировкой выигрывают у лучших по
обычному macro-F1; (2) поправка вероятностей реплик на доли классов
p'(c) ∝ p(c) · (1/3) / π(c) поднимает macro-F1 при равных долях ценой обычного.
Проверяется только на CV, скрытый тест для этого не используется.
""")
code("""
prior = pd.Series(y).value_counts(normalize=True)[REACTIONS]
weights = (1 / 3) / prior.to_numpy()
print("доли классов:", prior.round(3).to_dict(), " множители:", np.round(weights, 2))

def both(pred):
    return (np.mean([macro_f1(y[v], pred[v]) for _, _, v in iter_folds(df)]),
            np.mean([macro_f1(y[v], pred[v], w_eq[v]) for _, _, v in iter_folds(df)]))

stack_meta = variants["LR: SVM + Forest + LightGBM (hw2_stack)"]
plain, _ = S.oof_predict(df, L1T, *stack_meta)
corrected, _ = S.oof_predict(df, L1T, *stack_meta, weights=weights)
print("совпадение с CV hw2_stack:", (plain == res["hw2_stack"]["pred"]).mean())
rows = {n: both(res[n]["pred"]) for n in ["hw1_decision_2x", "hw1_score_stacking_lr", "hw1_grisha_909", "Stack_lr",
                                           "Forest_tfidf_300", "hw2_stack"]}
rows["hw2_stack + поправка на доли"] = both(corrected)
hyp = pd.DataFrame(rows, index=["macro-F1", "macro-F1 при равных долях"]).T
display(hyp)
ax = hyp.plot.barh(figsize=(8, 4), xlim=(0.6, 0.76)); ax.grid(axis="x", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "hypothesis.png", dpi=130); plt.show()

d2x = res["hw1_decision_2x"]["pred"]
by_fold = pd.DataFrame({
    "macro-F1: hw2_stack_prior": [macro_f1(y[v], corrected[v]) for _, _, v in iter_folds(df)],
    "macro-F1: hw1_decision_2x": [macro_f1(y[v], d2x[v]) for _, _, v in iter_folds(df)],
    "равные доли: hw2_stack_prior": [macro_f1(y[v], corrected[v], w_eq[v]) for _, _, v in iter_folds(df)],
    "равные доли: hw1_decision_2x": [macro_f1(y[v], d2x[v], w_eq[v]) for _, _, v in iter_folds(df)],
})
by_fold.loc["mean"] = by_fold.mean()
display(by_fold)
eq = pd.DataFrame({"delta": by_fold.iloc[:, 2] - by_fold.iloc[:, 3]})
paired_plot(eq, "hw2_stack_prior", "hw1_decision_2x (macro-F1 при равных долях)", FIG / "paired_delta_equal.png")
""")

# ---------------------------------------------------------------------------
md("""
## 11. Финальные модели

`python train_final.py` обучает `hw2_*` на всех 3000 диалогах и сохраняет в `artifacts/`;
`solution.ipynb` грузит их вместе с `hw1_*`.
""")
code("""
final = pd.DataFrame({"fit_seconds": json.loads(Path("artifacts/hw2_fit_seconds.json").read_text())})
final["file_mb"] = [Path(f"artifacts/{n}.joblib").stat().st_size / 2**20 for n in final.index]
final["cv_macro_f1"] = [f1({"hw2_random_forest": "Forest_tfidf_300", "hw2_hgb": "HGB", "hw2_lightgbm": "LightGBM_d6",
                            "hw2_stack": "hw2_stack"}[n]).mean() if n != "hw2_stack_prior" else hyp.loc["hw2_stack + поправка на доли", "macro-F1"]
                     for n in final.index]
final["cv_macro_f1_equal"] = [table.loc[{"hw2_random_forest": "Forest_tfidf_300", "hw2_hgb": "HGB", "hw2_lightgbm": "LightGBM_d6",
                                         "hw2_stack": "hw2_stack"}[n], "cv_macro_f1_equal"] if n != "hw2_stack_prior"
                              else hyp.loc["hw2_stack + поправка на доли", "macro-F1 при равных долях"] for n in final.index]
display(final)

import joblib
stack = joblib.load("artifacts/hw2_stack.joblib")
coef = pd.DataFrame(stack.meta_[-1].coef_, index=stack.meta_[-1].classes_,
                    columns=["svm_pos", "svm_neg", "forest_neg", "forest_neu", "forest_pos", "lgbm_neg", "lgbm_neu", "lgbm_pos"])
print("порог hw2_stack:", stack.threshold)
display(coef.round(2))
""")

md("""
## Итог

| | CV macro-F1 | при равных долях | лучше Decision_2x по фолдам |
|:--|--:|--:|:--|
| hw1_decision_2x (лучшая старая) | 0.7295 | 0.7103 | — |
| **hw2_stack** (лучшая новая) | **0.7330** | 0.6987 | 2 из 5 |
| **hw2_stack_prior** (то же + поправка на доли) | 0.7311 | **0.7335** | 3 из 5 / **5 из 5** |
| Forest_tfidf_300 (лучшее отдельное дерево) | 0.7160 | 0.6831 | 0 из 5 |
| LightGBM_d6 (лучший бустинг) | 0.7147 | 0.6905 | — |

1. Отдельные деревья и бустинг на репликах упираются в ~0.715: линейный SVM на 42k
   TF-IDF-признаках (Decision_2x) сильнее любого одиночного дерева.
2. Лучшая новая модель — стекинг на OOF по репликам из старой модели (SVM Decision_2x)
   и двух ансамблей. Она обходит лучшую старую по среднему CV macro-F1 (+0.0035), но
   прирост меньше разброса фолдов (±0.015): по сути это паритет. Вариант выбран из
   нескольких по внешней CV, оценка слегка оптимистична.
3. Стек по сводкам диалога (как у коллег) хуже стека по репликам: Decision_2x и деревья
   расходятся в выборе решающей реплики, а сводка диалога это теряет.
4. Веса классов в мета-модели (`balanced`) снижают macro-F1 на ~0.02–0.04, зато
   поднимают macro-F1 при равных долях — та же картина у стеков HW1.
5. Гипотеза подтверждена на CV: поправка вероятностей реплик на доли классов (без
   подбора) даёт `hw2_stack_prior`, который при равных долях лучше Decision_2x на всех
   5 фолдах (+0.023), сохраняя обычный macro-F1. Для скрытого теста с балансом 1:1:1
   это основной кандидат.
6. Главная ошибка всех моделей — пропущенная реакция (≈370 из ≈620 ошибок); около 400
   диалогов не распознаёт ни одна модель.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "experiments.ipynb")
print("experiments.ipynb:", len(cells), "ячеек")
