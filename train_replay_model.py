"""Модель для повторов: только признаки, которые извлекаем из любой .dem."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


folder = Path(__file__).resolve().parent
features = [
    "map", "phase", "time_since_phase_start", "bombsite", "defuse_in_progress",
    "t_dead", "ct_dead", "t_live_hp", "ct_live_hp", "ct_defuse_kits",
]
for side in ("t", "ct"):
    features += [
        f"{side}_{item}" for item in (
            "health_0_24", "health_25_49", "health_50_74", "health_75_100",
            "armor_0_24", "armor_25_49", "armor_50_74", "armor_75_100",
            "helmets", "fire_grenades", "smoke_grenades", "he_grenades",
            "flashbangs", "equipment_value",
        )
    ]
df = pd.read_parquet(folder / "data" / "cs2-round-states.parquet")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
assert set(train["demo_sha256"]).isdisjoint(test["demo_sha256"])

X_train = pd.get_dummies(train[features], dummy_na=True, dtype=int)
X_test = pd.get_dummies(test[features], dummy_na=True, dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
ct_reasons = {"bomb_defusal", "t_eliminated", "time_ran_out"}
y_train = train["round_end_reason"].isin(ct_reasons).astype(int)
y_test = test["round_end_reason"].isin(ct_reasons).astype(int)

models = {
    "Логистическая регрессия": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Случайный лес": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=10, random_state=42, n_jobs=-1
    ),
}
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
fit_idx, val_idx = next(splitter.split(X_train, y_train, groups=train["demo_sha256"]))
validation = {}
for name, candidate in models.items():
    candidate.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
    prob = candidate.predict_proba(X_train.iloc[val_idx])[:, 1]
    validation[name] = round(float(brier_score_loss(y_train.iloc[val_idx], prob)), 3)
name = min(validation, key=validation.get)
model = models[name]
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
metrics = {
    "selected": name,
    "validation_brier": validation,
    "test_accuracy": round(float(accuracy_score(y_test, prob >= 0.5)), 3),
    "test_brier": round(float(brier_score_loss(y_test, prob)), 3),
    "test_states": len(test),
    "test_matches": int(test["demo_sha256"].nunique()),
}
joblib.dump({"model": model, "features": features, "columns": X_train.columns.tolist(),
             "metrics": metrics}, folder / "replay_model.joblib")
(folder / "replay_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("Модель для демо:", metrics)
