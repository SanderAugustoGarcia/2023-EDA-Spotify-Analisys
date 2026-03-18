"""
Spotify 2023 — EDA Dashboard
Streamlit + Plotly · Visual identity Spotify
Deploy: https://streamlit.io/cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify 2023 · EDA",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Spotify theme CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
      background-color: #121212;
      color: #FFFFFF;
  }
  .stApp { background-color: #121212; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background-color: #000000 !important;
      border-right: 1px solid #282828;
  }
  section[data-testid="stSidebar"] * { color: #B3B3B3 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label { color: #FFFFFF !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
      background: #1E1E1E;
      border: 1px solid #282828;
      border-radius: 12px;
      padding: 16px 20px;
  }
  [data-testid="metric-container"] label { color: #B3B3B3 !important; font-size: 12px !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      color: #1DB954 !important;
      font-size: 28px !important;
      font-weight: 700 !important;
  }

  /* Headers */
  h1 { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: -1px; }
  h2 { color: #FFFFFF !important; font-weight: 700 !important; }
  h3 { color: #1DB954 !important; font-weight: 700 !important; font-size: 16px !important; }

  /* Insight boxes */
  .insight-box {
      background: #1E1E1E;
      border-left: 3px solid #1DB954;
      border-radius: 0 8px 8px 0;
      padding: 14px 18px;
      margin: 12px 0;
      font-size: 14px;
      line-height: 1.6;
      color: #B3B3B3;
  }
  .insight-box strong { color: #FFFFFF; }
  .insight-box .highlight { color: #1DB954; font-weight: 700; }
  .insight-warn { border-left-color: #E8563A; }
  .insight-warn .highlight { color: #E8563A; }
  .insight-amber { border-left-color: #F59B23; }
  .insight-amber .highlight { color: #F59B23; }
  .insight-purple { border-left-color: #9B72CF; }
  .insight-purple .highlight { color: #9B72CF; }

  /* Divider */
  hr { border-color: #282828 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #000000; border-radius: 8px; gap: 2px; }
  .stTabs [data-baseweb="tab"] {
      color: #B3B3B3 !important;
      background: transparent !important;
      border-radius: 6px !important;
      font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
      background: #1DB954 !important;
      color: #000000 !important;
      font-weight: 700 !important;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

  /* Selectbox / multiselect */
  .stSelectbox div[data-baseweb="select"] > div,
  .stMultiSelect div[data-baseweb="select"] > div {
      background-color: #282828 !important;
      border-color: #535353 !important;
      color: #FFFFFF !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #121212; }
  ::-webkit-scrollbar-thumb { background: #535353; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly template Spotify ───────────────────────────────────────────────────
_t = go.layout.Template()
_t.layout.paper_bgcolor = "#121212"
_t.layout.plot_bgcolor  = "#1E1E1E"
_t.layout.font          = dict(family="DM Sans", color="#B3B3B3", size=12)
_t.layout.title         = dict(font=dict(color="#FFFFFF", size=15, family="DM Sans"), x=0.01)
_t.layout.xaxis         = dict(gridcolor="#282828", linecolor="#282828", tickcolor="#535353", zerolinecolor="#282828")
_t.layout.yaxis         = dict(gridcolor="#282828", linecolor="#282828", tickcolor="#535353", zerolinecolor="#282828")
_t.layout.colorway      = ["#1DB954","#E8563A","#F59B23","#9B72CF","#4FC3F7","#B3B3B3"]
_t.layout.legend        = dict(bgcolor="#1E1E1E", bordercolor="#282828", borderwidth=1, font=dict(color="#B3B3B3"))
_t.layout.margin        = dict(l=50, r=30, t=50, b=50)
_t.layout.hoverlabel    = dict(bgcolor="#282828", bordercolor="#535353", font=dict(color="#FFFFFF", size=12))
pio.templates["spotify"] = _t
SPOTIFY_TEMPLATE = "spotify"

# ── Cores ─────────────────────────────────────────────────────────────────────
GREEN  = "#1DB954"
CORAL  = "#E8563A"
AMBER  = "#F59B23"
PURPLE = "#9B72CF"
BLUE   = "#4FC3F7"
GRAY   = "#B3B3B3"

ERA_COLORS = {
    "Pré-2000":  "#B3B3B3",
    "2000s":     "#9B72CF",
    "2010s":     "#4FC3F7",
    "2020–2021": "#1DB954",
    "2022–2023": "#E8563A",
}

# ── ETL ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_raw = pd.read_csv("spotify-2023.csv", encoding="utf-8", encoding_errors="replace")

    # Remover linha corrompida
    bad = pd.to_numeric(df_raw["streams"], errors="coerce").isna()
    df  = df_raw[~bad].copy()
    df["streams"] = pd.to_numeric(df["streams"])

    # Converter numéricos
    num_cols = [
        "artist_count","released_year","released_month","released_day",
        "in_spotify_playlists","in_spotify_charts","in_apple_playlists",
        "in_apple_charts","in_deezer_playlists","in_deezer_charts","in_shazam_charts",
        "bpm","danceability_%","valence_%","energy_%","acousticness_%",
        "instrumentalness_%","liveness_%","speechiness_%",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Tratar nulos
    df["key"] = df["key"].replace("", np.nan).fillna("Unknown")
    df["in_shazam_charts"] = df["in_shazam_charts"].fillna(0)

    # Features
    df["streams_M"]  = df["streams"] / 1e6
    df["era"]        = pd.cut(df["released_year"],
                              bins=[0,1999,2009,2019,2021,2023],
                              labels=["Pré-2000","2000s","2010s","2020–2021","2022–2023"])
    df["collab"]     = df["artist_count"].apply(lambda x: "Colaboração" if x > 1 else "Solo")
    df["era"]        = df["era"].astype(str)
    return df

df = load_data()

# ── Sidebar — filtros ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Spotify 2023")
    st.markdown("<p style='color:#535353;font-size:12px'>EDA Dashboard · 952 músicas</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Filtros")

    eras_all   = ["Pré-2000","2000s","2010s","2020–2021","2022–2023"]
    eras_sel   = st.multiselect("Era de lançamento", eras_all, default=eras_all)

    modes_all  = sorted(df["mode"].dropna().unique().tolist())
    modes_sel  = st.multiselect("Modo", modes_all, default=modes_all)

    keys_all   = sorted(df["key"].dropna().unique().tolist())
    keys_sel   = st.multiselect("Tonalidade (key)", keys_all, default=keys_all)

    year_min, year_max = int(df["released_year"].min()), int(df["released_year"].max())
    year_range = st.slider("Ano de lançamento", year_min, year_max, (year_min, year_max))

    st.markdown("---")
    st.markdown("<p style='color:#535353;font-size:11px'>Dataset: Kaggle · Most Streamed Spotify Songs 2023</p>", unsafe_allow_html=True)

# Aplicar filtros
mask = (
    df["era"].isin(eras_sel) &
    df["mode"].isin(modes_sel) &
    df["key"].isin(keys_sel) &
    df["released_year"].between(year_range[0], year_range[1])
)
dff = df[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎵 Spotify 2023 — Análise Exploratória de Dados")
st.markdown(f"<p style='color:#B3B3B3;margin-top:-12px'>Pipeline completo ETL → EDA · <strong style='color:#1DB954'>{len(dff)}</strong> músicas seleccionadas</p>", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

s_sorted    = np.sort(dff["streams"].dropna().values)
cum_s       = np.cumsum(s_sorted) / s_sorted.sum() if len(s_sorted) else np.array([0])
cum_p       = np.arange(1, len(s_sorted)+1) / len(s_sorted) if len(s_sorted) else np.array([0])
gini        = float(1 - 2 * np.trapezoid(cum_s, cum_p)) if len(s_sorted) > 1 else 0
idx_10      = int(0.9 * len(s_sorted))
share_top10 = float((1 - cum_s[idx_10]) * 100) if len(s_sorted) > 1 else 0

k1.metric("Músicas", f"{len(dff):,}")
k2.metric("Streams total", f"{dff['streams'].sum()/1e9:.1f}B")
k3.metric("Mediana streams", f"{dff['streams_M'].median():.0f}M")
k4.metric("Gini (desigualdade)", f"{gini:.2f}")
k5.metric("Top 10% = X% streams", f"{share_top10:.0f}%")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Q1 · Distribuição",
    "📅 Q2 · Sazonalidade",
    "🕰️ Q3 · Músicas antigas",
    "🤝 Q4 · Colaborações",
    "🏆 Top músicas",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Distribuição de streams
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Q1 — Quão desigual é a distribuição de streams?")

    col1, col2 = st.columns(2)

    # Histograma log
    with col1:
        log_s = np.log10(dff["streams"].clip(lower=1))
        fig = go.Figure(template=SPOTIFY_TEMPLATE)
        fig.add_trace(go.Histogram(
            x=log_s, nbinsx=40,
            marker_color=GREEN, opacity=0.85,
            hovertemplate="log₁₀(streams): %{x:.2f}<br>Músicas: %{y}<extra></extra>",
            name="Distribuição"
        ))
        fig.add_vline(x=float(np.log10(dff["streams"].median())),
                      line_dash="dash", line_color=CORAL, line_width=2,
                      annotation_text=f"Mediana {dff['streams_M'].median():.0f}M",
                      annotation_font_color=CORAL)
        fig.add_vline(x=float(np.log10(dff["streams"].mean())),
                      line_dash="dot", line_color=AMBER, line_width=2,
                      annotation_text=f"Média {dff['streams_M'].mean():.0f}M",
                      annotation_font_color=AMBER, annotation_position="top left")
        fig.update_layout(title="Distribuição em log₁₀ (streams)",
                          xaxis_title="log₁₀(Streams)", yaxis_title="Nº de músicas",
                          showlegend=False, height=370)
        st.plotly_chart(fig, use_container_width=True)

    # Curva de Lorenz
    with col2:
        fig2 = go.Figure(template=SPOTIFY_TEMPLATE)
        fig2.add_trace(go.Scatter(
            x=cum_p*100, y=cum_s*100,
            mode="lines", line=dict(color=GREEN, width=2.5),
            fill="tonexty", fillcolor="rgba(29,185,84,0.1)",
            name="Curva de Lorenz",
            hovertemplate="Top %{x:.1f}% das músicas<br>= %{y:.1f}% dos streams<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=[0,100], y=[0,100],
            mode="lines", line=dict(color="#535353", width=1.2, dash="dash"),
            name="Igualdade perfeita",
        ))
        fig2.add_annotation(
            x=75, y=30,
            text=f"<b>Top 10% = {share_top10:.0f}% streams</b><br>Gini = {gini:.2f}",
            showarrow=True, arrowhead=2, ax=0, ay=-40,
            bgcolor="#1E1E1E", bordercolor=GREEN, borderwidth=1,
            font=dict(color=GREEN, size=12),
        )
        fig2.update_layout(title="Curva de Lorenz",
                           xaxis_title="% músicas (acumulado)",
                           yaxis_title="% streams (acumulado)",
                           height=370)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Insight:</strong> A distribuição segue uma <span class="highlight">lei de potência</span> —
        a média (<span class="highlight">{:.0f}M</span>) é quase o dobro da mediana (<span class="highlight">{:.0f}M</span>).
        O Gini de <span class="highlight">{:.2f}</span> indica desigualdade moderada-alta:
        o top 10% das músicas concentra <span class="highlight">{:.0f}%</span> de todos os streams.
        A transformação log₁₀ aproxima a distribuição de uma normal — relevante para modelação futura.
    </div>
    """.format(dff["streams_M"].mean(), dff["streams_M"].median(), gini, share_top10),
    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Sazonalidade
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Q2 — Existe sazonalidade nos lançamentos musicais?")

    df_recent = dff[dff["released_year"] >= 2022].copy()
    month_names = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

    m_count   = df_recent.groupby("released_month").size().reindex(range(1,13), fill_value=0)
    m_streams = df_recent.groupby("released_month")["streams_M"].median().reindex(range(1,13), fill_value=0)

    col1, col2 = st.columns(2)

    with col1:
        colors_vol = [CORAL if v == m_count.max() else GREEN for v in m_count.values]
        fig = go.Figure(template=SPOTIFY_TEMPLATE)
        fig.add_trace(go.Bar(
            x=month_names, y=m_count.values,
            marker_color=colors_vol, opacity=0.9,
            hovertemplate="%{x}: <b>%{y} músicas</b><extra></extra>",
            text=m_count.values, textposition="outside",
            textfont=dict(color="#FFFFFF", size=11),
        ))
        fig.add_hline(y=m_count.mean(), line_dash="dash", line_color="#535353",
                      annotation_text=f"Média {m_count.mean():.0f}",
                      annotation_font_color="#535353")
        fig.update_layout(title="Volume de lançamentos por mês (2022–2023)",
                          yaxis_title="Nº de músicas", showlegend=False, height=370)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        colors_str = [AMBER if v == m_streams.max() else GREEN for v in m_streams.values]
        fig2 = go.Figure(template=SPOTIFY_TEMPLATE)
        fig2.add_trace(go.Bar(
            x=month_names, y=m_streams.values,
            marker_color=colors_str, opacity=0.9,
            hovertemplate="%{x}: mediana <b>%{y:.0f}M streams</b><extra></extra>",
            text=[f"{v:.0f}M" for v in m_streams.values],
            textposition="outside",
            textfont=dict(color="#FFFFFF", size=11),
        ))
        fig2.update_layout(title="Streams mediana por mês de lançamento",
                           yaxis_title="Streams mediana (M)", showlegend=False, height=370)
        st.plotly_chart(fig2, use_container_width=True)

    best_vol = month_names[m_count.idxmax()-1]
    best_str = month_names[m_streams.idxmax()-1]
    worst    = month_names[m_count.idxmin()-1]

    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Insight:</strong> <span class="highlight">{best_vol}</span> lidera em volume de lançamentos ({m_count.max()} músicas)
        — a indústria abre o ano forte. <strong>{worst}</strong> é o mês mais calmo ({m_count.min()} músicas — férias da indústria).
        Mas músicas lançadas em <span class="highlight">{best_str}</span> têm a maior mediana de streams,
        sugerindo um <span class="highlight">trade-off entre volume e visibilidade</span>:
        lançar com menos concorrência pode valer mais do que lançar na "época alta".
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Músicas antigas
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Q3 — Músicas antigas ainda dominam os charts de 2023?")

    era_order = ["Pré-2000","2000s","2010s","2020–2021","2022–2023"]
    era_order_f = [e for e in era_order if e in dff["era"].unique()]

    col1, col2 = st.columns(2)

    with col1:
        era_counts = dff["era"].value_counts().reindex(era_order_f, fill_value=0)
        fig = go.Figure(template=SPOTIFY_TEMPLATE)
        fig.add_trace(go.Bar(
            x=era_counts.index.tolist(),
            y=era_counts.values,
            marker_color=[ERA_COLORS.get(e, GREEN) for e in era_counts.index],
            opacity=0.9,
            hovertemplate="%{x}: <b>%{y} músicas</b><extra></extra>",
            text=era_counts.values, textposition="outside",
            textfont=dict(color="#FFFFFF", size=11),
        ))
        fig.update_layout(title="Nº de músicas por era de lançamento",
                          yaxis_title="Nº de músicas", showlegend=False, height=370)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(template=SPOTIFY_TEMPLATE)
        for era in era_order_f:
            sub = dff[dff["era"] == era]
            if len(sub) == 0:
                continue
            color = ERA_COLORS.get(era, GREEN)
            fig2.add_trace(go.Box(
                y=sub["streams_M"],
                name=era,
                marker_color=color,
                line_color=color,
                fillcolor=f"rgba{tuple(list(bytes.fromhex(color.lstrip('#'))) + [40])}",
                boxmean=True,
                hovertemplate=f"<b>{era}</b><br>Streams: %{{y:.0f}}M<extra></extra>",
            ))
        fig2.update_layout(title="Distribuição de streams por era (boxplot)",
                           yaxis_title="Streams (M)", showlegend=False, height=370)
        st.plotly_chart(fig2, use_container_width=True)

    # Tabela resumo por era
    era_summary = (
        dff.groupby("era")["streams_M"]
        .agg(["count","median","mean","max"])
        .reindex(era_order_f)
        .rename(columns={"count":"Músicas","median":"Mediana (M)","mean":"Média (M)","max":"Máx (M)"})
        .round(0)
        .astype({"Músicas": int})
    )
    st.dataframe(era_summary, use_container_width=True)

    st.markdown("""
    <div class="insight-box insight-purple">
        ⚠️ <strong>Viés de sobrevivência:</strong> As músicas dos <span class="highlight">anos 2000</span> têm a maior mediana —
        mas isso não significa que músicas antigas são "melhores". As únicas músicas antigas que chegam
        aos charts de 2023 são os <span class="highlight">maiores clássicos de sempre</span>.
        As milhares de músicas "normais" dessa era simplesmente não estão neste dataset.
        O catálogo histórico é um activo de <span class="highlight">alto valor passivo</span> para as editoras.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Colaborações
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Q4 — Colaborações geram mais streams do que solos?")

    col1, col2, col3 = st.columns(3)

    with col1:
        counts = dff["collab"].value_counts().reindex(["Solo","Colaboração"], fill_value=0)
        fig = go.Figure(template=SPOTIFY_TEMPLATE)
        fig.add_trace(go.Bar(
            x=counts.index.tolist(), y=counts.values,
            marker_color=[GREEN, CORAL], opacity=0.9,
            text=[f"{v}<br>({v/len(dff)*100:.0f}%)" for v in counts.values],
            textposition="outside", textfont=dict(color="#FFFFFF", size=12),
            hovertemplate="%{x}: <b>%{y} músicas</b><extra></extra>",
        ))
        fig.update_layout(title="Volume no dataset", yaxis_title="Nº de músicas",
                          showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        medians = dff.groupby("collab")["streams_M"].median().reindex(["Solo","Colaboração"], fill_value=0)
        fig2 = go.Figure(template=SPOTIFY_TEMPLATE)
        fig2.add_trace(go.Bar(
            x=medians.index.tolist(), y=medians.values,
            marker_color=[GREEN, CORAL], opacity=0.9,
            text=[f"{v:.0f}M" for v in medians.values],
            textposition="outside", textfont=dict(color="#FFFFFF", size=13, family="DM Sans"),
            hovertemplate="%{x}: mediana <b>%{y:.0f}M</b><extra></extra>",
        ))
        fig2.update_layout(title="Streams mediana", yaxis_title="Streams mediana (M)",
                           showlegend=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        dff["artist_count_capped"] = dff["artist_count"].clip(upper=5)
        by_n = dff.groupby("artist_count_capped")["streams_M"].median()
        n_labels = {1:"1 (solo)",2:"2",3:"3",4:"4",5:"5+"}
        x_labels = [n_labels.get(int(k), str(k)) for k in by_n.index]
        colors_n = [GREEN, BLUE, AMBER, CORAL, PURPLE][:len(by_n)]
        fig3 = go.Figure(template=SPOTIFY_TEMPLATE)
        fig3.add_trace(go.Bar(
            x=x_labels, y=by_n.values,
            marker_color=colors_n, opacity=0.9,
            text=[f"{v:.0f}M" for v in by_n.values],
            textposition="outside", textfont=dict(color="#FFFFFF", size=11),
            hovertemplate="Nº artistas %{x}: mediana <b>%{y:.0f}M</b><extra></extra>",
        ))
        fig3.update_layout(title="Mediana por nº de artistas",
                           xaxis_title="Nº de artistas", yaxis_title="Streams mediana (M)",
                           showlegend=False, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    solo_m   = dff[dff["collab"]=="Solo"]["streams_M"].median()
    collab_m = dff[dff["collab"]=="Colaboração"]["streams_M"].median()
    diff_pct = (solo_m/collab_m - 1)*100 if collab_m > 0 else 0

    st.markdown(f"""
    <div class="insight-box insight-warn">
        💡 <strong>Resultado contra-intuitivo:</strong> Solos têm mediana
        <span class="highlight">{diff_pct:.0f}% superior</span> às colaborações ({solo_m:.0f}M vs {collab_m:.0f}M).
        A hipótese mais plausível: artistas com bases de fãs consolidadas
        (Taylor Swift, Bad Bunny, The Weeknd) lançam maioritariamente solos.
        <strong>O factor determinante é a base de fãs do artista principal</strong>,
        não o número de artistas na faixa.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Top músicas
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🏆 Top músicas")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        n_top = st.slider("Número de músicas", 5, 50, 20)
        sort_by = st.selectbox("Ordenar por", [
            "streams", "in_spotify_playlists", "in_spotify_charts",
            "danceability_%", "energy_%", "bpm"
        ])

    top_df = (
        dff.nlargest(n_top, sort_by)
        [[
            "track_name","artist(s)_name","released_year","streams_M",
            "in_spotify_playlists","bpm","danceability_%","energy_%","mode","key","collab","era"
        ]]
        .rename(columns={
            "track_name":            "Música",
            "artist(s)_name":        "Artista(s)",
            "released_year":         "Ano",
            "streams_M":             "Streams (M)",
            "in_spotify_playlists":  "Playlists Spotify",
            "bpm":                   "BPM",
            "danceability_%":        "Danceability",
            "energy_%":              "Energy",
            "mode":                  "Modo",
            "key":                   "Tonalidade",
            "collab":                "Tipo",
            "era":                   "Era",
        })
        .reset_index(drop=True)
    )
    top_df.index += 1
    top_df["Streams (M)"] = top_df["Streams (M)"].round(0).astype(int)

    st.dataframe(
        top_df.style.background_gradient(subset=["Streams (M)"], cmap="Greens"),
        use_container_width=True,
        height=600,
    )

    # Bar chart top 10
    top10 = dff.nlargest(10, "streams_M")
    fig = go.Figure(template=SPOTIFY_TEMPLATE)
    fig.add_trace(go.Bar(
        x=top10["streams_M"],
        y=top10["track_name"] + "  —  " + top10["artist(s)_name"],
        orientation="h",
        marker_color=GREEN, opacity=0.9,
        hovertemplate="<b>%{y}</b><br>%{x:.0f}M streams<extra></extra>",
        text=[f"{v:.0f}M" for v in top10["streams_M"]],
        textposition="outside", textfont=dict(color=GREEN, size=11),
    ))
    fig.update_layout(
        title="Top 10 músicas por streams",
        xaxis_title="Streams (M)",
        yaxis=dict(autorange="reversed"),
        height=420, showlegend=False,
        margin=dict(l=280, r=80),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#535353;font-size:12px'>"
    "🎵 Spotify 2023 EDA · Dataset: Kaggle · Construído com Streamlit + Plotly"
    "</p>",
    unsafe_allow_html=True
)
