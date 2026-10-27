"""Streamlit dashboard for the Beijing PM2.5 forecasts.

    streamlit run dashboard/app.py

Reads the files written by ``python train.py`` from ``artifacts/h<hours>/``.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).resolve().parent.parent / "artifacts"

# Fixed colour per entity (never by rank), from the validated reference palette.
COLORS = {"actual": "#8a8984", "xgboost": "#2a78d6", "lstm": "#eb6834", "hybrid": "#1baf7a", "persistence": "#b4b2aa"}
LABELS = {"actual": "Actual", "xgboost": "XGBoost", "lstm": "LSTM", "hybrid": "Hybrid", "persistence": "Persistence"}
# China ambient air quality (GB 3095-2012) 24 h PM2.5 limits, ug/m3
THRESHOLDS = [(35, "Grade I limit (35)"), (75, "Grade II limit (75)")]

st.set_page_config(page_title="Beijing PM2.5 Forecast", page_icon="🌫️", layout="wide")


@st.cache_data
def load(horizon_dir):
    d = ART / horizon_dir
    metrics = json.loads((d / "metrics.json").read_text())
    preds = pd.read_csv(d / "test_predictions.csv", parse_dates=["forecast_made_at"])
    preds["valid_at"] = preds["forecast_made_at"] + pd.Timedelta(hours=metrics["horizon_hours"])
    return metrics, preds


def scores(y, p):
    e = y - p
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e**2).mean()), "R²": 1 - (e**2).sum() / ((y - y.mean()) ** 2).sum()}


def style(fig, height=380, unified=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="x unified" if unified else "closest",
                      legend=dict(orientation="h", y=1.1, x=0), template="plotly")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


horizons = sorted([p.name for p in ART.glob("h*") if (p / "metrics.json").exists()], key=lambda s: int(s[1:]))
if not horizons:
    st.error("No artifacts found. Run `python train.py --horizon 24 --out-dir artifacts/h24` first.")
    st.stop()

st.title("Beijing PM2.5 forecast")
st.caption("Hold-out evaluation of XGBoost and LSTM forecasters on 12 monitoring stations (data: Mar 2013 – Feb 2017).")

with st.sidebar:
    st.header("Filters")
    horizon_dir = st.selectbox(
        "Forecast horizon", horizons, format_func=lambda h: f"{int(h[1:])} h" + (" (7 days)" if h == "h168" else "")
    )
    metrics, preds = load(horizon_dir)
    models = [m for m in ("xgboost", "lstm", "hybrid") if f"pred_{m}" in preds]
    chosen = st.multiselect("Models", models, default=[metrics.get("best_model", models[0])], format_func=LABELS.get)
    station = st.selectbox("Station", ["All stations (average)"] + sorted(preds["station"].unique()))
    lo, hi = preds["valid_at"].min().date(), preds["valid_at"].max().date()
    rng = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    smooth = st.radio("Resolution", ["Daily mean", "Hourly"], horizontal=True)

df = preds if station.startswith("All") else preds[preds["station"] == station]
if len(rng) == 2:
    df = df[(df["valid_at"].dt.date >= rng[0]) & (df["valid_at"].dt.date <= rng[1])]
if df.empty or not chosen:
    st.warning("Nothing to show for these filters.")
    st.stop()

tab_fc, tab_cmp, tab_err, tab_drv, tab_data = st.tabs(["Forecast", "Model comparison", "Errors", "Drivers", "Data"])

with tab_fc:
    ref = chosen[0]
    s = scores(df["actual"], df[f"pred_{ref}"])
    base = scores(df["actual"], df["current_pm25"])
    c = st.columns(4)
    c[0].metric(f"MAE · {LABELS[ref]}", f"{s['MAE']:.1f} µg/m³", f"{s['MAE'] - base['MAE']:+.1f} vs persistence", delta_color="inverse")
    c[1].metric("RMSE", f"{s['RMSE']:.1f} µg/m³", f"{s['RMSE'] - base['RMSE']:+.1f} vs persistence", delta_color="inverse")
    c[2].metric("R²", f"{s['R²']:.2f}", f"{s['R²'] - base['R²']:+.2f} vs persistence")
    c[3].metric("Mean actual PM2.5", f"{df['actual'].mean():.0f} µg/m³")

    cols = ["actual"] + [f"pred_{m}" for m in chosen]
    plot = df.set_index("valid_at")[cols].groupby(level=0).mean()
    if smooth == "Daily mean":
        plot = plot.resample("D").mean()
    fig = go.Figure()
    fig.add_scatter(x=plot.index, y=plot["actual"], name="Actual", line=dict(color=COLORS["actual"], width=2))
    for m in chosen:
        fig.add_scatter(x=plot.index, y=plot[f"pred_{m}"], name=LABELS[m], line=dict(color=COLORS[m], width=2))
    for v, label in THRESHOLDS:
        fig.add_hline(y=v, line_dash="dot", line_color="rgba(128,128,128,0.6)", annotation_text=label,
                      annotation_position="top left", annotation_font_size=10)
    fig.update_yaxes(title="PM2.5 (µg/m³)")
    st.plotly_chart(style(fig, 420), width="stretch")
    st.caption(f"Each point is plotted at the time being forecast ({metrics['horizon_hours']} h after the forecast was made).")

with tab_cmp:
    rows = []
    for name in [m for m in ("xgboost", "lstm", "hybrid") if m in metrics] + ["persistence", "same_hour_yesterday"]:
        rows.append({"Model": LABELS.get(name, "Same hour yesterday"), **{k: metrics[name][k] for k in ("MAE", "RMSE", "R2")}})
    tbl = pd.DataFrame(rows)
    st.subheader("Hold-out metrics (all stations, full test period)")
    left, right = st.columns(2)
    for col, metric in zip((left, right), ("MAE", "RMSE")):
        colors = [COLORS.get(n.lower(), "#b4b2aa") for n in tbl["Model"]]
        fig = go.Figure(go.Bar(x=tbl["Model"], y=tbl[metric], marker_color=colors,
                               text=tbl[metric].round(1), textposition="outside"))
        fig.update_layout(title=f"{metric} (µg/m³, lower is better)", showlegend=False)
        fig.update_yaxes(rangemode="tozero")
        col.plotly_chart(style(fig, 320, unified=False), width="stretch")
    st.dataframe(tbl.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "R2": "{:.3f}"}), hide_index=True, width="stretch")
    st.info("The LSTM tends to win on MAE but lose on RMSE: it predicts typical days well and underestimates pollution spikes. "
            "Both beat persistence, but skill is low at long horizons because future weather is not an input.")

with tab_err:
    ref = chosen[0]
    d = df.assign(err=df[f"pred_{ref}"] - df["actual"])
    l, r = st.columns(2)
    fig = go.Figure(go.Histogram(x=d["err"].clip(-300, 300), nbinsx=60, marker_color=COLORS[ref]))
    fig.update_layout(title=f"Forecast error, {LABELS[ref]} (predicted − actual)", showlegend=False)
    fig.update_xaxes(title="µg/m³ (clipped at ±300)")
    l.plotly_chart(style(fig, 340, unified=False), width="stretch")
    samp = d.sample(min(len(d), 4000), random_state=0)
    fig = go.Figure(go.Scattergl(x=samp["actual"], y=samp[f"pred_{ref}"], mode="markers",
                                 marker=dict(color=COLORS[ref], size=4, opacity=0.35), name="forecasts"))
    mx = float(max(samp["actual"].max(), samp[f"pred_{ref}"].max()))
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="rgba(128,128,128,0.7)", dash="dash"), name="perfect")
    fig.update_layout(title="Predicted vs actual")
    fig.update_xaxes(title="Actual")
    fig.update_yaxes(title="Predicted")
    r.plotly_chart(style(fig, 340, unified=False), width="stretch")

    st.subheader("Error by station")
    by = preds.assign(ae=(preds[f"pred_{ref}"] - preds["actual"]).abs()).groupby("station")["ae"].mean().sort_values()
    fig = go.Figure(go.Bar(x=by.values, y=by.index, orientation="h", marker_color=COLORS[ref],
                           text=by.round(1), textposition="outside"))
    fig.update_layout(title=f"MAE by station, {LABELS[ref]} (µg/m³)", showlegend=False)
    st.plotly_chart(style(fig, 380, unified=False), width="stretch")

with tab_drv:
    imp = pd.Series(metrics["top_features"]).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=COLORS["xgboost"]))
    fig.update_layout(title="XGBoost: top features by importance", showlegend=False)
    st.plotly_chart(style(fig, 460, unified=False), width="stretch")
    st.caption("Importance is spread thinly across many lag/rolling features: no single variable dominates, "
               "which fits the low long-range skill.")

with tab_data:
    show = df[["forecast_made_at", "valid_at", "station", "actual", "current_pm25"] + [f"pred_{m}" for m in models]].round(1)
    st.dataframe(show, hide_index=True, width="stretch", height=420)
    st.download_button("Download CSV", show.to_csv(index=False), file_name=f"pm25_{horizon_dir}_forecasts.csv", mime="text/csv")
