"""Обучает финальные модели hw2_* на всех 3000 диалогах и сохраняет в artifacts/."""

import json
import time
import warnings

import joblib

from experiments import FINAL
from splits import load_data

warnings.filterwarnings("ignore")
df = load_data()
seconds = {}
for name, make in FINAL.items():
    start = time.perf_counter()
    model = make().fit(df)
    seconds[name] = round(time.perf_counter() - start, 1)
    joblib.dump(model, f"artifacts/{name}.joblib", compress=3)
    print(name, seconds[name], "с")
json.dump(seconds, open("artifacts/hw2_fit_seconds.json", "w"), indent=1)
