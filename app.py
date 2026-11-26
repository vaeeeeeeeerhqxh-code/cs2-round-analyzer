"""Первый интерфейс: реальные тестовые состояния и прогноз нашей модели."""

import base64
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


folder = Path(__file__).resolve().parent
st.set_page_config(page_title="CS2 Round Analyzer", page_icon="🎯", layout="wide")
st.markdown("""
<style>
    .stApp {background: #101722; color: #eaf0f7;}
    .block-container {max-width: 1200px; padding-top: 2rem;}
    .hero {padding: 1.2rem 1.5rem; border: 1px solid #2b3c53;
           border-radius: 18px; background: #172335; margin-bottom: 1.5rem;}
    .hero h1 {margin: 0 0 .25rem; font-size: 2rem;}
    .hero p {margin: 0; color: #aebcd0;}
</style>
""", unsafe_allow_html=True)

replay_folder = folder / "data" / "local_replays"
replay_ready = (replay_folder / "replay.json").exists() and (replay_folder / "radar.png").exists()
if replay_ready:
    view = st.segmented_control("Раздел", ["Повтор матча", "Анализ модели"], default="Повтор матча")
    if view == "Повтор матча":
        replay = json.loads((replay_folder / "replay.json").read_text(encoding="utf-8"))
        radar = base64.b64encode((replay_folder / "radar.png").read_bytes()).decode("ascii")
        payload = json.dumps(replay, ensure_ascii=False).replace("<", "\\u003c")
        html = (folder / "replay_ui.html").read_text(encoding="utf-8")
        html = html.replace("__REPLAY_JSON__", payload).replace("__RADAR_BASE64__", radar)
        components.html(html, height=1050, scrolling=False)
        st.stop()


@st.cache_data
def read_states():
    return pd.read_parquet(folder / "data" / "cs2-round-states.parquet")


@st.cache_resource
def read_model():
    return joblib.load(folder / "model.joblib")


df = read_states()
bundle = read_model()
test = df[df["split"] == "test"].copy()

# Кодирование должно быть точно таким же, как в train.py.
X = pd.get_dummies(test[bundle["features"]], dummy_na=True, dtype=int)
X = X.reindex(columns=bundle["columns"], fill_value=0)
X = X.fillna(bundle["fill_values"])
test["ct_probability"] = bundle["model"].predict_proba(X)[:, 1]

st.markdown("""
<div class="hero"><h1>CS2 Round Analyzer</h1>
<p>Как меняется оценка победы в раунде · первый прототип на реальных данных</p></div>
""", unsafe_allow_html=True)

matches = sorted(test["demo_filename"].unique())
default_match = "aurora-vs-spirit-m1-mirage.dem"
match = st.selectbox(
    "Тестовый матч", matches,
    index=matches.index(default_match) if default_match in matches else 0,
)
states = test[test["demo_filename"] == match].sort_values("tick").reset_index(drop=True)

# В данном выпуске набора каждый раунд содержит ровно 4 сохранённых состояния.
# Проверяем это по целевой метке, чтобы не смешать соседние раунды.
assert len(states) % 4 == 0
assert (states.groupby(states.index // 4)["round_end_reason"].nunique() == 1).all()
round_count = len(states) // 4
round_number = st.selectbox("Раунд в наборе данных", range(1, round_count + 1))
round_states = states.iloc[(round_number - 1) * 4:round_number * 4].copy()
round_states["step"] = range(1, 5)

if "step" not in st.session_state:
    st.session_state.step = 1
left, right = st.columns([1, 1])
with left:
    if st.button("← Предыдущее состояние", disabled=st.session_state.step == 1):
        st.session_state.step -= 1
with right:
    if st.button("Следующее состояние →", disabled=st.session_state.step == 4):
        st.session_state.step += 1
step = st.slider("Состояние внутри раунда", 1, 4, key="step")
row = round_states.iloc[step - 1]
ct_p = float(row["ct_probability"])

col_chart, col_state = st.columns([1.6, 1], gap="large")
with col_chart:
    st.subheader(f"Раунд {round_number}: оценка шансов")
    chart_data = pd.DataFrame({
        "Состояние": list(range(1, 5)) * 2,
        "Вероятность": list(round_states["ct_probability"]) +
                       list(1 - round_states["ct_probability"]),
        "Команда": ["CT"] * 4 + ["T"] * 4,
    })
    chart = alt.Chart(chart_data).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("Состояние:O", sort=[1, 2, 3, 4]),
        y=alt.Y("Вероятность:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        color=alt.Color("Команда:N", scale=alt.Scale(
            domain=["CT", "T"], range=["#55bfff", "#f5ae51"]
        )),
        tooltip=["Состояние", "Команда", alt.Tooltip("Вероятность:Q", format=".1%")],
    ).properties(height=300)
    st.altair_chart(chart, width="stretch")
    if step > 1:
        previous = float(round_states.iloc[step - 2]["ct_probability"])
        change = (ct_p - previous) * 100
        st.info(f"Между состояниями {step-1} и {step} оценка CT изменилась "
                f"на {change:+.1f} процентного пункта. События между ними "
                "пока не известны: для этого нужна исходная .dem.")

with col_state:
    st.subheader(f"Состояние {step} из 4")
    a, b = st.columns(2)
    a.metric("Шанс CT", f"{ct_p:.1%}")
    b.metric("Шанс T", f"{1-ct_p:.1%}")
    st.progress(ct_p)
    a, b = st.columns(2)
    a.metric("CT живы", 5 - int(row["ct_dead"]))
    b.metric("T живы", 5 - int(row["t_dead"]))
    a, b = st.columns(2)
    a.metric("Здоровье CT", int(row["ct_live_hp"]))
    b.metric("Здоровье T", int(row["t_live_hp"]))
    phase = "Бомба установлена" if row["phase"] == "bomb_planted" else "До установки бомбы"
    st.write(f"**Фаза:** {phase}")
    if pd.notna(row["bombsite"]):
        st.write(f"**Плент:** {row['bombsite']}")
    if st.toggle("Показать фактический исход раунда"):
        reason = row["round_end_reason"]
        winner = "CT" if reason in {"bomb_defusal", "t_eliminated", "time_ran_out"} else "T"
        st.success(f"Победила команда {winner} · {reason}")

with st.expander("Откуда взялись эти числа?"):
    st.write("Это четыре выбранных автором датасета состояния раунда, а не непрерывное "
             "воспроизведение демки. Вероятности рассчитаны сохранённой моделью "
             "`model.joblib`. Положение игроков и точные события появятся после "
             "обработки исходной `.dem`.")
    st.write(f"**Карта:** {row['map']} · **tick:** {int(row['tick'])} · "
             f"**время текущей фазы:** {row['time_since_phase_start']:.1f} с")
