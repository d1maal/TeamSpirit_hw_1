"""Графики для отчёта."""

import matplotlib.pyplot as plt
import numpy as np

BASE, BEST, OTHER, NEW = "#d62728", "#2ca02c", "#9e9e9e", "#1f77b4"


def colors(names, base, best):
    return [BASE if n == base else BEST if n == best else OTHER if n.startswith("hw1_") else NEW for n in names]


def compare_plot(table, folds, base, best, path=None):
    """Строка = модель. Панели: CV macro-F1 с точками фолдов, размер, время обучения и предсказания."""
    table = table.sort_values("cv_macro_f1")
    names, y = list(table.index), np.arange(len(table))
    c = colors(names, base, best)
    panels = [("cv_macro_f1", "CV macro-F1"), ("size_mb", "размер модели, МБ"),
              ("fit_seconds", "обучение, с"), ("predict_seconds", "предсказание 600 диалогов, с")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 0.38 * len(names) + 1.5), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        ax.barh(y, table[col], color=c, alpha=0.85)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    ax = axes[0]
    for i, n in enumerate(names):
        ax.scatter(folds[n], [i] * len(folds[n]), color="black", s=8, zorder=3)
    ax.axvline(table.loc[base, "cv_macro_f1"], color=BASE, ls="--", lw=1)
    ax.set_xlim(0.5, 0.78)
    axes[0].set_yticks(y, names)
    for ax in axes[1:3]:
        ax.set_xscale("log")
    fig.suptitle(f"Красный — лучшая старая модель ({base}), зелёный — лучшая новая ({best}); точки — фолды")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()


def paired_plot(delta, a, b, path=None):
    """Разница macro-F1 a - b по фолдам."""
    d = delta["delta"].iloc[:-1].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(d)), d, color=[BEST if v > 0 else BASE for v in d])
    ax.axhline(d.mean(), color="black", ls="--", lw=1, label=f"среднее {d.mean():+.4f}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("фолд")
    ax.set_ylabel("Δ macro-F1")
    ax.set_title(f"{a} − {b}: лучше на {(d > 0).sum()} из {len(d)} фолдов")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
