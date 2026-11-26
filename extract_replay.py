"""Извлечь один настоящий раунд CS2 для локального воспроизведения.

Запуск: python extract_replay.py "C:\\path\\to\\match.dem" [--round 3]
Личные данные из демки сохраняются только в data/local_replays/ (папка в .gitignore).
"""

import argparse
import json
import shutil
from pathlib import Path

import polars as pl
from awpy import Demo, data, plot

ASSET_VERSION = 2000950  # Закрепляем версию карты, чтобы не зависеть от GitHub API.

parser = argparse.ArgumentParser(description="Извлечь позиции и события из CS2 .dem")
parser.add_argument("demo", type=Path, help="Путь к файлу .dem")
parser.add_argument("--round", type=int, help="Номер раунда; по умолчанию выберем интересный")
args = parser.parse_args()

if not args.demo.is_file():
    parser.error(f"Файл не найден: {args.demo}")

demo = Demo(args.demo)
map_name = demo.header["map_name"]
tick_rate = demo.tick_rate
rounds = demo.rounds.filter(
    pl.col("freeze_end_tick").is_not_null(),
    pl.col("end_tick") > pl.col("freeze_end_tick"),
)
kills = demo.kills
bomb = demo.bomb

if args.round is not None:
    chosen = rounds.filter(pl.col("round_num") == args.round)
    if chosen.is_empty():
        parser.error(f"Раунд {args.round} не найден")
    chosen = chosen.row(0, named=True)
else:
    # Для первой демонстрации выбираем раунд с установкой бомбы и несколькими убийствами.
    candidates = []
    for item in rounds.iter_rows(named=True):
        start, end = item["freeze_end_tick"], item["end_tick"]
        kill_count = kills.filter(pl.col("tick").is_between(start, end)).height
        plant_count = bomb.filter(
            pl.col("tick").is_between(start, end), pl.col("event") == "finish_plant"
        ).height
        candidates.append((int(plant_count > 0), min(kill_count, 5), item))
    if not candidates:
        parser.error("В демке нет завершённых раундов")
    chosen = max(candidates, key=lambda x: (x[0], x[1]))[2]

start, end = chosen["freeze_end_tick"], chosen["end_tick"]
snapshots = demo.snapshots(seconds=1.0, start_tick=start, end_tick=end)
if snapshots.is_empty():
    parser.error("Awpy не нашёл позиции игроков в выбранном раунде")

frames = []
for tick, players in snapshots.group_by("tick", maintain_order=True):
    tick = tick[0] if isinstance(tick, tuple) else tick
    entries = []
    for player in players.iter_rows(named=True):
        if player["x"] is None or player["y"] is None:
            continue
        px, py = plot.world_to_pixel(map_name, (player["x"], player["y"]), version=ASSET_VERSION)
        entries.append({
            "id": str(player["steamid"] or player["name"]),
            "name": player["name"] or "Игрок",
            "side": "CT" if player["side"] == "counter-terrorist" else "T",
            "x": round(px, 1), "y": round(py, 1),
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

events = []
for kill in kills.filter(pl.col("tick").is_between(start, end)).iter_rows(named=True):
    events.append({
        "time": round((kill["tick"] - start) / tick_rate, 2),
        "type": "kill",
        "text": f"{kill['attacker_name'] or 'Неизвестно'} → {kill['victim_name'] or 'игрок'}",
    })
for action in bomb.filter(pl.col("tick").is_between(start, end)).iter_rows(named=True):
    if action["event"] in {"finish_plant", "defuse"}:
        text = (f"Бомба установлена на {action['bombsite']}" if action["event"] == "finish_plant"
                else "Бомба обезврежена")
        events.append({
            "time": round((action["tick"] - start) / tick_rate, 2),
            "type": action["event"],
            "site": action["bombsite"] if action["event"] == "finish_plant" else None,
            "text": text,
        })
events.sort(key=lambda item: item["time"])

out = Path(__file__).resolve().parent / "data" / "local_replays"
out.mkdir(parents=True, exist_ok=True)
radar = data.radar_path(map_name, version=ASSET_VERSION)
shutil.copyfile(radar, out / "radar.png")
replay = {
    "map": map_name,
    "round": chosen["round_num"],
    "duration": round((end - start) / tick_rate, 2),
    "winner": chosen["winner_side"],
    "frames": frames,
    "events": events,
}
(out / "replay.json").write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")
print(f"Карта: {map_name}; раунд: {chosen['round_num']}; кадров: {len(frames)}; "
      f"событий: {len(events)}")
print(f"Сохранено: {out / 'replay.json'}")
