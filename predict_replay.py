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
    outcome_time = replay.get("outcome_time", replay["duration"])
    last_active_probability = float(probabilities[0])
    for frame, probability in zip(replay["frames"], probabilities):
        if frame["time"] < outcome_time:
            last_active_probability = float(probability)
        # После исхода раунда прогноз больше не обновляем.
        frame["ct_probability"] = round(last_active_probability, 4)

    active = [frame for frame in replay["frames"] if frame["time"] < outcome_time]
    measured_events = []
    for event in replay["events"]:
        before = next((frame for frame in reversed(active) if frame["time"] < event["time"]), None)
        after = next((frame for frame in active if frame["time"] >= event["time"]), None)
        if before is None or after is None or after["time"] - before["time"] > 2.1:
            continue
        delta_pp = round(100 * (after["ct_probability"] - before["ct_probability"]), 1)
        event["delta_pp"] = delta_pp
        event["after_time"] = after["time"]
        measured_events.append(event)
    biggest = max(measured_events, key=lambda event: abs(event["delta_pp"]), default=None)
    replay["summary"] = {
        "initial_ct_probability": active[0]["ct_probability"] if active else None,
        "biggest_event": ({key: biggest[key] for key in ("time", "after_time", "text", "type", "delta_pp")}
                          if biggest else None),
    }
    return replay
