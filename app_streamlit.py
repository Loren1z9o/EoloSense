"""Painel EoloSense: compara os modelos e mostra, evento a evento, quando e por que o alerta disparou.

Na raiz do projeto, depois de treinar (Comum/src/treinar_modelos.py):
    ./.venv/Scripts/python.exe -m pip install -r requirements-app.txt
    ./.venv/Scripts/python.exe -m streamlit run app_streamlit.py

O painel só LÊ as pastas outputs/. A aba "Como o modelo decide" também lê o CSV do
evento em data/ para refazer a previsão de um instante; sem o dataset, ela avisa e segue.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Comum/src"))

MODELS = {  # chave: (pasta de resultados, nome, cor fixa por modelo)
    "kmeans": ("K-means/outputs", "K-means", "#2a78d6"),
    "arvore": ("Arvore-de-Decisao/outputs", "Árvore de Decisão", "#eb6834"),
    "ensembles": ("Ensembles/outputs", "Ensembles", "#1baf7a"),
    "svm": ("SVM/outputs", "SVM (SVR)", "#eda100"),
    "linear": ("Regressao-Linear/outputs", "Regressão Linear", "#e87ba4"),
}
MUTED, CRITICAL, WINDOW = "#52514e", "#d03b3b", "rgba(120,120,110,0.14)"
KMEANS_RUN = "detection_balanced"

st.set_page_config(page_title="EoloSense", page_icon="🌬️", layout="wide")


# --- Leitura dos resultados -----------------------------------------------------------
def runs() -> list[str]:
    found = set()
    for key, (folder, _, _) in MODELS.items():
        base = ROOT / folder
        if key != "kmeans" and base.exists():
            found |= {p.name for p in base.iterdir() if (p / "events.csv").exists()}
    return sorted(found, key=lambda r: (r != "principal", r))


def run_dir(key: str, run: str) -> Path:
    folder = ROOT / MODELS[key][0]
    return folder / (KMEANS_RUN if key == "kmeans" else run)


def available(run: str) -> list[str]:
    keys = [k for k in MODELS if k != "kmeans" and (run_dir(k, run) / "events.csv").exists()]
    if (run_dir("kmeans", run) / "events.csv").exists():
        keys.insert(0, "kmeans")
    return keys


@st.cache_data(show_spinner=False)
def events_table(key: str, run: str) -> pd.DataFrame:
    root = run_dir(key, run)
    frozen = json.loads((root / "frozen_config.json").read_text(encoding="utf-8"))
    events = pd.read_csv(root / "events.csv", dtype={"event": str})
    return events.loc[events.config.eq(frozen["config"])].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def windows(run: str) -> pd.DataFrame:
    """Rótulo e janela anotada de cada evento (gravados pelos modelos supervisionados)."""
    for key in available(run):
        table = events_table(key, run)
        if "event_start" in table:
            return table[["farm", "event", "split", "label", "event_start", "event_end"]].drop_duplicates()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def predictions(key: str, run: str, farm: str, event: str) -> pd.DataFrame | None:
    path = run_dir(key, run) / ("" if key == "kmeans" else "eventos") / f"farm_{farm}_event_{event}" / "predictions.csv"
    return pd.read_csv(path, parse_dates=["time_stamp"]) if path.exists() else None


def report(key: str, run: str, farm: str, event: str) -> dict:
    path = run_dir(key, run) / "eventos" / f"farm_{farm}_event_{event}" / "report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pct(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.0%}"


def decision(row) -> str:
    if row.anomaly:
        return "✅ Verdadeiro positivo: avisou dentro da janela" if row.detected else "❌ Falso negativo: falha sem aviso na janela"
    return "❌ Falso positivo: avisou num evento normal" if row.detected else "✅ Verdadeiro negativo: nenhum aviso"


# --- Interface ---------------------------------------------------------------------------
st.title("EoloSense · detecção precoce de falhas em turbinas eólicas")
st.caption("CARE to Compare · cada modelo aprende a operação normal da turbina e avisa quando os sensores se afastam dela.")

all_runs = runs()
if not all_runs:
    st.warning("Nenhum resultado encontrado. Treine primeiro: `python Comum/src/treinar_modelos.py --jobs 4` "
               "(veja Comum/README.md).")
    st.stop()

with st.sidebar:
    st.header("Execução")
    run = st.selectbox("Resultados", all_runs, help="Subpasta outputs/<run> criada pelo treino")
    keys = available(run)
    st.markdown("**Modelos encontrados**")
    for key in keys:
        st.markdown(f"<span style='color:{MODELS[key][2]}'>■</span> {MODELS[key][1]}", unsafe_allow_html=True)

tab_compare, tab_event, tab_theory = st.tabs(["📊 Comparação", "📈 Evento a evento", "🧠 Como o modelo decide"])

# --- Aba 1: comparação ---------------------------------------------------------------------
with tab_compare:
    path = ROOT / "Comum/outputs" / run / "comparacao.csv"
    if not path.exists():
        st.info("Comparação ainda não gerada. Rode `python Comum/src/treinar_modelos.py --compare-only`.")
    else:
        table = pd.read_csv(path)
        st.subheader("Eventos reservados (nenhuma escolha usou esses eventos)")
        cols = st.columns(len(table))
        for col, row in zip(cols, table.itertuples()):
            col.metric(row.modelo, f"{row.balanced_accuracy:.0%}", help="Acurácia balanceada")
            col.caption(f"Sens. {pct(row.sensitivity)} · Espec. {pct(row.specificity)} · "
                        f"{row.false_notifications_per_30_days:.1f} avisos falsos/30 dias")
        color_by_name = {MODELS[k][1].split(" (")[0]: MODELS[k][2] for k in MODELS}
        measures = {"sensitivity": "Sensibilidade", "specificity": "Especificidade",
                    "precision": "Precisão", "balanced_accuracy": "Acurácia balanceada"}
        figure = go.Figure()
        for row in table.itertuples():
            name = row.modelo.split(" (")[0]
            values = [getattr(row, m) for m in measures]
            figure.add_bar(name=row.modelo, x=list(measures.values()), y=values,
                           marker_color=color_by_name.get(name, MUTED),
                           hovertemplate="%{x}: %{y:.0%}<extra>" + row.modelo + "</extra>")
        # Metas só sobre as medidas a que se referem.
        for position, (goal, text) in enumerate([(.7, "meta 70%"), (.8, "meta 80%")]):
            figure.add_shape(type="line", x0=position - .45, x1=position + .45, y0=goal, y1=goal,
                             line=dict(color=MUTED, dash="dot", width=2))
            figure.add_annotation(x=position, y=goal, text=text, showarrow=False, yshift=10,
                                  font=dict(color=MUTED, size=11), bgcolor="rgba(255,255,255,.8)")
        figure.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_range=[0, 1.05], height=420,
                             legend_orientation="h", margin=dict(t=30, b=10))
        st.plotly_chart(figure, width="stretch")
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Sensibilidade = falhas com aviso novo dentro da janela anotada. Especificidade = eventos normais sem "
                   "nenhum aviso. Métricas por evento, não por registro de 10 minutos; não é o CARE score oficial.")

# --- Aba 2: evento a evento ----------------------------------------------------------------
with tab_event:
    info = windows(run)
    if info.empty:
        st.info("Janelas dos eventos indisponíveis nesta execução.")
    else:
        c1, c2, c3 = st.columns([1.2, 1, 2])
        key = c1.selectbox("Modelo", keys, format_func=lambda k: MODELS[k][1], index=min(1, len(keys) - 1))
        split = c2.radio("Conjunto", ["evaluation", "selection"], horizontal=True,
                         format_func=lambda s: "reservados" if s == "evaluation" else "desenvolvimento")
        options = info.loc[info.split.eq(split)].sort_values(["farm", "event"], key=lambda c: c.astype(str).str.zfill(4))
        label = {r.Index: f"Fazenda {r.farm} · evento {r.event} · {r.label}" for r in options.itertuples()}
        choice = c3.selectbox("Evento", list(label), format_func=label.get)
        meta = options.loc[choice]
        data = predictions(key, run, meta.farm, meta.event)
        if data is None:
            st.warning("Esse modelo não tem previsões para o evento escolhido.")
        else:
            row = events_table(key, run).query("farm == @meta.farm and event == @meta.event").iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Decisão", "Aviso" if row.detected else "Sem aviso")
            m2.metric("Avisos emitidos", int(data.notification.sum()))
            m3.metric("Registros em alarme", f"{int(data.alarm.sum())}/{len(data)}")
            m4.metric("Dias de previsão", f"{data.time_stamp.sub(data.time_stamp.min()).max().days}")
            st.markdown(f"**{decision(row)}**")

            figure = go.Figure()
            if meta.label == "anomaly":
                figure.add_vrect(x0=meta.event_start, x1=meta.event_end, fillcolor=WINDOW, line_width=0,
                                 annotation_text="janela anotada da anomalia", annotation_position="bottom left")
            active = data.incident_active.astype(bool)
            if active.any():
                figure.add_scatter(x=data.time_stamp, y=np.where(active, data.score.max() * 1.05, np.nan),
                                   mode="lines", line=dict(color=CRITICAL, width=6), opacity=.35,
                                   name="incidente aberto", hoverinfo="skip")
            hover = "%{x|%d/%m %H:%M}<br>escore %{y:.2f}"
            custom = None
            if "top_sensor" in data:
                custom = data.top_sensor.fillna("—")
                hover += "<br>sensor mais desviado: %{customdata}"
            figure.add_scatter(x=data.time_stamp, y=data.score, mode="lines", name="escore",
                               line=dict(color=MODELS[key][2], width=1.6), customdata=custom,
                               hovertemplate=hover + "<extra></extra>")
            figure.add_scatter(x=data.time_stamp, y=data.threshold, mode="lines", name="limiar da calibração",
                               line=dict(color=MUTED, dash="dash", width=1.4), hoverinfo="skip")
            warn = data.loc[data.notification.astype(bool)]
            figure.add_scatter(x=warn.time_stamp, y=warn.score, mode="markers", name="aviso emitido",
                               marker=dict(color=CRITICAL, size=11, line=dict(color="white", width=2)),
                               hovertemplate="AVISO %{x|%d/%m %H:%M}<extra></extra>")
            figure.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", y=-.18),
                                 margin=dict(t=30, b=40), yaxis_title="escore de anomalia")
            st.plotly_chart(figure, width="stretch")

            if "top_sensor" in data and data.alarm.any():
                details = report(key, run, meta.farm, meta.event)
                names = details.get("target_descriptions", {})
                counts = data.loc[data.alarm.astype(bool), "top_sensor"].value_counts()
                st.markdown("**Sensores que mais desviaram durante os alarmes**")
                st.dataframe(pd.DataFrame(dict(sensor=counts.index, descricao=[names.get(s, "") for s in counts.index],
                                               registros=counts.to_numpy())), hide_index=True)
            details = report(key, run, meta.farm, meta.event)
            if details:
                with st.expander("Qualidade da regressão neste evento (calibração, dados normais)"):
                    st.write(f"Configuração escolhida na validação: **{details['chosen']}**")
                    st.dataframe(pd.DataFrame(details["regression"]["per_target"]).round(3), hide_index=True)
                    st.write("Entradas:", ", ".join(details["inputs"] + details["engineered"]))

# --- Aba 3: teoria aplicada a um instante -------------------------------------------------------
with tab_theory:
    st.markdown(
        "Os modelos supervisionados não classificam *falha* diretamente. Eles respondem: **qual temperatura cada "
        "componente deveria ter agora, se a turbina estivesse saudável?** O passo a passo abaixo refaz a conta de um "
        "instante real e confere com o `predict()` do modelo salvo.")
    supervised = [k for k in keys if k != "kmeans"]
    info = windows(run)
    if not supervised or info.empty:
        st.info("Treine ao menos um modelo supervisionado para usar esta aba.")
        st.stop()
    c1, c2 = st.columns(2)
    key = c1.selectbox("Modelo ", supervised, format_func=lambda k: MODELS[k][1])
    label = {r.Index: f"Fazenda {r.farm} · evento {r.event} · {r.label}" for r in info.itertuples()}
    choice = c2.selectbox("Evento ", list(label), format_func=label.get)
    meta = info.loc[choice]
    folder = run_dir(key, run) / "eventos" / f"farm_{meta.farm}_event_{meta.event}"
    manifest = pd.read_csv(run_dir(key, run) / "manifest.csv", dtype={"event": str})
    source = Path(manifest.query("farm == @meta.farm and event == @meta.event").path.iloc[0])
    if not source.exists() or not (folder / "model.joblib").exists():
        st.info(f"Para esta aba é preciso o CSV original do evento ({source.name}) em data/ e o model.joblib.")
        st.stop()

    import nbm

    @st.cache_resource(show_spinner="Carregando modelo e dados do evento…")
    def load(folder: str, source: str):
        bundle = joblib.load(Path(folder) / "model.joblib")
        frame, _, _ = nbm.load_event(Path(source))
        layout = nbm.Layout(bundle["inputs"], bundle["engineered"], [], {})
        full = nbm.add_engineered(frame, layout)
        return bundle, full.loc[nbm.split_rows(full)[1]].reset_index(drop=True)

    bundle, prediction = load(str(folder), str(source))
    scores = nbm.score_frame(bundle, prediction, meta.farm)
    valid = scores.index[scores.eligible].to_numpy()
    default = int(valid[scores.loc[valid, "score"].to_numpy().argmax()])
    position = st.slider("Instante da seção de previsão (padrão: maior escore)", 0, len(prediction) - 1, default)
    row = prediction.iloc[[position]]
    st.markdown(f"**{row.time_stamp.iloc[0]:%d/%m/%Y %H:%M}** · modelo: `{bundle['chosen']}`")

    st.markdown("**1. Condições de operação (entradas)**")
    st.dataframe(row[bundle["x_columns"]].T.rename(columns={row.index[0]: "valor"}).round(3), width="content")

    observed, predicted, raw = nbm.predict_z(bundle, row)
    expected = bundle["y_scaler"].inverse_transform(predicted)[0]
    check = bundle["model"].predict(row[bundle["x_columns"]].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    agree = np.allclose(check, predicted)
    st.markdown("**2. Temperatura esperada × real** e **3. resíduo padronizado** = (real − esperado) / desvio na validação")
    residual = (observed[0] - predicted[0]) / bundle["sigma"]
    st.dataframe(pd.DataFrame({
        "sensor": bundle["targets"],
        "descrição": [bundle["target_descriptions"].get(t, "") for t in bundle["targets"]],
        "real": raw[0], "esperado (predict)": expected, "diferença": raw[0] - expected,
        "resíduo padronizado": residual}).round(3), hide_index=True, width="stretch")
    st.caption(("✅ Conferido: o cálculo acima é idêntico ao `model.predict()`." if agree
                else "⚠️ O cálculo não bateu com `model.predict()`."))

    st.markdown("**4. Escore** = média móvel de 1 h do resíduo de cada sensor → maior valor absoluto")
    q = json.loads((run_dir(key, run) / "frozen_config.json").read_text(encoding="utf-8"))["quantile"]
    threshold = bundle["calibration_thresholds"][q]
    s = scores.iloc[position]
    a, b, c = st.columns(3)
    a.metric("Escore neste instante", "—" if pd.isna(s.score) else f"{s.score:.2f}")
    b.metric(f"Limiar (percentil {q:.1%} da calibração)", f"{threshold:.2f}")
    c.metric("Sensor que mais desviou", s.top_sensor or "—")
    st.markdown(
        f"Um registro acima do limiar é só uma **excedência**. O aviso exige persistência: a fração de excedências na "
        f"janela causal precisa passar do valor congelado no desenvolvimento, e um episódio aberto só gera um aviso.")

    model = bundle["model"].steps[-1][1]
    if key == "arvore" and hasattr(model, "decision_path"):
        st.markdown("**Caminho na árvore para este instante**")
        x = bundle["model"][:-1].transform(row[bundle["x_columns"]].apply(pd.to_numeric, errors="coerce").to_numpy(float))
        tree, nodes = model.tree_, model.decision_path(x).indices
        lines = []
        for node in nodes:
            if tree.children_left[node] == -1:
                lines.append(f"→ folha {node}: média de {tree.n_node_samples[node]} registros de treino normal")
                continue
            name, cut = bundle["x_columns"][tree.feature[node]], tree.threshold[node]
            side = "≤" if x[0, tree.feature[node]] <= cut else ">"
            lines.append(f"{name} = {x[0, tree.feature[node]]:.3f} {side} {cut:.3f}")
        st.code("\n".join(lines), language=None)
        leaf = tree.value[nodes[-1]][:, 0]
        st.caption("✅ O valor da folha é exatamente o `predict()` (em desvios-padrão de cada alvo)."
                   if np.allclose(leaf, predicted[0]) else "⚠️ A folha não bateu com o `predict()`.")
    elif bundle.get("importances"):
        st.markdown("**Importância das entradas no modelo**")
        st.bar_chart(pd.Series(bundle["importances"]).head(10), horizontal=True, color=MODELS[key][2])
