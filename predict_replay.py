"""Прогноз CT для каждого кадра повтора по тем же признакам, что при обучении."""

import pandas as pd


def predict_replay(replay, bundle):
    rows = []
    plants = [event for event in replay["events"] if event["type"] == "finish_plant"]
    for frame in replay["frames"]:
        plant = next((e for e in reversed(plants) if e["time"] <= frame["time"]), None)
        ct = [p for p in frame["players"] if p["side"] == "CT"]
        t = [p for p in frame["players"] if p["side"] == "T"]
        row = {
            "map": replay["map"],
            "phase": "bomb_planted" if plant else "bomb_not_planted",
            "time_since_phase_start": frame["time"] - plant["time"] if plant else frame["time"],
            "bombsite": plant["site"] if plant else None,
            "defuse_in_progress": any(p["defusing"] for p in ct if p["hp"] > 0),
            "ct_defuse_kits": sum(p["defuse_kit"] for p in ct if p["hp"] > 0),
        }
        for side, players in (("t", t), ("ct", ct)):
            alive = [p for p in players if p["hp"] > 0]
            row[f"{side}_dead"] = 5 - len(alive)
            row[f"{side}_live_hp"] = sum(p["hp"] for p in alive)
            for lo, hi in ((0, 24), (25, 49), (50, 74), (75, 100)):
                row[f"{side}_health_{lo}_{hi}"] = sum(lo <= p["hp"] <= hi for p in alive)
                row[f"{side}_armor_{lo}_{hi}"] = sum(lo <= p["armor"] <= hi for p in alive)
            row[f"{side}_helmets"] = sum(p["helmet"] for p in alive)
            for field in ("fire_grenades", "smoke_grenades", "he_grenades", "flashbangs", "equipment_value"):
                row[f"{side}_{field}"] = sum(p[field] for p in alive)
        rows.append(row)
    X = pd.get_dummies(pd.DataFrame(rows)[bundle["features"]], dummy_na=True, dtype=int)
    X = X.reindex(columns=bundle["columns"], fill_value=0)
    probabilities = bundle["model"].predict_proba(X)[:, 1]
    for frame, probability in zip(replay["frames"], probabilities):
        frame["ct_probability"] = round(float(probability), 4)
    return replay
