"""Первый прототип: прогноз победителя раунда по текущему состоянию CS2."""

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


folder = Path(__file__).resolve().parent
data_file = folder / "data" / "cs2-round-states.parquet"
manifest = json.loads((folder / "data" / "dataset-manifest.json").read_text())
actual_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
if actual_hash != manifest["sha256"]:
    raise ValueError("Датасет не совпадает с файлом автора")

df = pd.read_parquet(data_file)
print("Состояний:", len(df))
print("Матчей:", df["demo_sha256"].nunique())
print("Классы:", df["round_end_reason"].value_counts().to_dict())

# Исход раунда известен только после его завершения: это правильный ответ, а не признак.
ct_reasons = {"bomb_defusal", "t_eliminated", "time_ran_out"}
df["ct_win"] = df["round_end_reason"].isin(ct_reasons).astype(int)

train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()
assert set(train["demo_sha256"]).isdisjoint(test["demo_sha256"])

# Оставляем только признаки состояния. Идентификаторы, tick и исход удаляем.
not_features = [
    "demo_id", "demo_filename", "demo_sha256", "tick", "split",
    "round_end_reason", "ct_win",
]
features = [name for name in df.columns if name not in not_features]
X_train = pd.get_dummies(train[features], dummy_na=True, dtype=int)
X_test = pd.get_dummies(test[features], dummy_na=True, dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
fill_values = X_train.median(numeric_only=True).fillna(0)
X_train = X_train.fillna(fill_values)
X_test = X_test.fillna(fill_values)
y_train, y_test = train["ct_win"], test["ct_win"]

models = {
    "Простой ориентир": DummyClassifier(strategy="prior"),
    "Логистическая регрессия": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "Случайный лес": RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    results[name] = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "brier": float(brier_score_loss(y_test, probability)),
    }
    print(f"{name}: accuracy={results[name]['accuracy']:.3f}, "
          f"Brier={results[name]['brier']:.3f}")
    # Эту понятную модель сохраняем для будущего интерфейса.
    if name == "Логистическая регрессия":
        joblib.dump({
            "model": model,
            "features": features,
            "columns": X_train.columns.tolist(),
            "fill_values": fill_values,
        }, folder / "model.joblib")
        pd.DataFrame({
            "demo_filename": test["demo_filename"].to_numpy(),
            "tick": test["tick"].to_numpy(),
            "actual_ct_win": y_test.to_numpy(),
            "predicted_ct_probability": probability,
        }).to_csv(folder / "predictions.csv", index=False)

(folder / "results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
)
plt.figure(figsize=(7, 4))
plt.bar(results.keys(), [value["accuracy"] for value in results.values()])
plt.ylabel("Доля верных прогнозов")
plt.ylim(0, 1)
plt.xticks(rotation=12, ha="right")
plt.tight_layout()
plt.savefig(folder / "model_comparison.png", dpi=160)
print("Сохранены model.joblib, results.json, predictions.csv, model_comparison.png")
