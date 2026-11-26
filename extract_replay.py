"""Извлечь раунды CS2 для локального повтора.

Запуск: python extract_replay.py "C:\\path\\to\\match.dem" [--round 3]
Личные данные остаются в data/local_replays/ (эта папка исключена из Git).
"""

import argparse
import json
import shutil
from pathlib import Path

import polars as pl
from awpy import Demo, data


ASSET_VERSION = 2000950  # Закреплённая версия карты.
parser = argparse.ArgumentParser(description="Извлечь позиции и события из CS2 .dem")
parser.add_argument("demo", type=Path, help="Путь к файлу .dem")
parser.add_argument("--round", type=int, help="Извлечь только один раунд; без этого — все")
args = parser.parse_args()
if not args.demo.is_file():
    parser.error(f"Файл не найден: {args.demo}")

demo = Demo(args.demo)
map_name = demo.header["map_name"]
tick_rate = demo.tick_rate
rounds = demo.rounds.filter(
    pl.col("freeze_end_tick").is_not_null(),
    pl.col("end_tick") > pl.col("freeze_end_tick"),
).sort("round_num")
if args.round is not None:
    rounds = rounds.filter(pl.col("round_num") == args.round)
if rounds.is_empty():
    parser.error("В демке нет выбранного завершённого раунда")

kills, bomb = demo.kills, demo.bomb
meta = data.map_data(ASSET_VERSION)[map_name]
out = Path(__file__).resolve().parent / "data" / "local_replays"
out.mkdir(parents=True, exist_ok=True)
shutil.copyfile(data.radar_path(map_name, version=ASSET_VERSION), out / "radar.png")


def export_round(item):
    start, end = item["freeze_end_tick"], item["end_tick"]
    snapshots = demo.snapshots(seconds=1.0, start_tick=start, end_tick=end)
    if snapshots.is_empty():
        return None

    frames = []
    for key, players in snapshots.group_by("tick", maintain_order=True):
        tick = key[0] if isinstance(key, tuple) else key
        entries = []
        for player in players.iter_rows(named=True):
            if player["x"] is None or player["y"] is None:
                continue
            entries.append({
                "id": str(player["steamid"] or player["name"]),
                "name": player["name"] or "Игрок",
                "side": "CT" if player["side"] == "counter-terrorist" else "T",
                "x": round((player["x"] - meta["pos_x"]) / meta["scale"], 1),
                "y": round((meta["pos_y"] - player["y"]) / meta["scale"], 1),
                "hp": int(player["health"] or 0),
                "armor": int(player["armor"] or 0),
                "bomb": bool(player["has_bomb"]),
                "helmet": bool(player["has_helmet"]),
                "defuse_kit": bool(player["has_defuser"]),
                "defusing": bool(player["is_defusing"]),
                "equipment_value": int(player["equipment_value"] or 0),
                "fire_grenades": int(player["fire_grenades"] or 0),
                "smoke_grenades": int(player["smoke_grenades"] or 0),
                "he_grenades": int(player["he_grenades"] or 0),
                "flashbangs": int(player["flashbangs"] or 0),
            })
        frames.append({"time": round((tick - start) / tick_rate, 2), "players": entries})

    round_kills = kills.filter(pl.col("tick").is_between(start, end))
    round_bomb = bomb.filter(pl.col("tick").is_between(start, end))
    events = [{
        "time": round((kill["tick"] - start) / tick_rate, 2),
        "type": "kill",
        "text": f"{kill['attacker_name'] or 'Неизвестно'} → {kill['victim_name'] or 'игрок'}",
    } for kill in round_kills.iter_rows(named=True)]
    for action in round_bomb.iter_rows(named=True):
        if action["event"] in {"finish_plant", "defuse"}:
            events.append({
                "time": round((action["tick"] - start) / tick_rate, 2),
                "type": action["event"],
                "site": action["bombsite"] if action["event"] == "finish_plant" else None,
                "text": (f"Бомба установлена на {action['bombsite']}"
                         if action["event"] == "finish_plant" else "Бомба обезврежена"),
            })
    events.sort(key=lambda event: event["time"])
    replay = {
        "map": map_name, "round": int(item["round_num"]),
        "duration": round((end - start) / tick_rate, 2),
        "winner": item["winner_side"], "frames": frames, "events": events,
    }
    filename = f"round_{item['round_num']}.json"
    (out / filename).write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")
    print(f"Раунд {item['round_num']}: {len(frames)} кадров, {len(events)} событий")
    return {"round": int(item["round_num"]), "file": filename,
            "duration": replay["duration"], "events": len(events),
            "has_plant": any(e["type"] == "finish_plant" for e in events)}


index = [result for item in rounds.iter_rows(named=True)
         if (result := export_round(item)) is not None]
if not index:
    parser.error("Awpy не нашёл позиции игроков в выбранных раундах")
default = max(index, key=lambda item: (item["has_plant"], min(item["events"], 5)))
(out / "index.json").write_text(json.dumps({"map": map_name, "rounds": index,
                                           "default_round": default["round"]}, ensure_ascii=False),
                                encoding="utf-8")
print(f"Готово: {len(index)} раундов из {map_name}. Откройте Streamlit.")
