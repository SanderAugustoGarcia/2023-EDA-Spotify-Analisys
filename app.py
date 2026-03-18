"""
Spotify 2023 — EDA Dashboard
Streamlit + Plotly · Estilo editorial premium
Módulos: EDA · Correlações · Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Spotify 2023 · EDA",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0a0a; color: #e8e6df; }
.stApp { background-color: #0a0a0a; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background-color: #111111 !important; border-right: 1px solid #1a1a1a; padding-top: 0 !important; }
section[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] button[title="Collapse sidebar"] { display: none !important; }

/* ── Sidebar logo ── */
.sidebar-logo { background: linear-gradient(135deg, #1DB954 0%, #158a3e 100%); padding: 28px 24px 22px; margin-bottom: 8px; }
.sidebar-logo .logo-text { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; color: #000; letter-spacing: -0.5px; line-height: 1; }
.sidebar-logo .logo-sub { font-size: 11px; color: rgba(0,0,0,0.6); margin-top: 4px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── Module selector ── */
.mod-selector { padding: 16px; border-bottom: 1px solid #1a1a1a; margin-bottom: 4px; }
.mod-label { font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #444; margin-bottom: 10px; }
.mod-btn { display: block; width: 100%; text-align: left; background: transparent; border: 1px solid #1a1a1a; border-radius: 10px; padding: 10px 14px; margin-bottom: 6px; cursor: pointer; transition: all .15s; }
.mod-btn:hover { border-color: #2a2a2a; background: #161616; }
.mod-btn.active { border-color: #1DB954; background: rgba(29,185,84,0.06); }
.mod-btn .mod-name { font-size: 13px; font-weight: 600; color: #e8e6df; }
.mod-btn.active .mod-name { color: #1DB954; }
.mod-btn .mod-desc { font-size: 10px; color: #444; margin-top: 2px; }

/* ── Sidebar filters ── */
section[data-testid="stSidebar"] label { color: #666 !important; font-size: 10px !important; font-weight: 600 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div { background-color: #1a1a1a !important; border-color: #2a2a2a !important; color: #e8e6df !important; border-radius: 8px !important; }
section[data-testid="stSidebar"] p { color: #555 !important; font-size: 12px !important; }
[data-testid="stSlider"] [role="slider"] { background-color: #1DB954 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div:nth-child(3) { background-color: #1DB954 !important; }

/* ── Hero ── */
.hero { padding: 48px 0 36px; border-bottom: 1px solid #1a1a1a; margin-bottom: 40px; }
.hero-eyebrow { display: inline-flex; align-items: center; gap: 8px; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 100px; padding: 5px 14px; font-size: 11px; font-weight: 600; color: #1DB954; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 20px; }
.hero-eyebrow::before { content: ''; width: 6px; height: 6px; background: #1DB954; border-radius: 50%; display: inline-block; }
.hero-title { font-family: 'Syne', sans-serif; font-size: clamp(36px, 5vw, 64px); font-weight: 800; color: #ffffff; line-height: 1.0; letter-spacing: -2px; margin: 0 0 16px; }
.hero-title span { color: #1DB954; }
.hero-sub { font-size: 16px; color: #666; font-weight: 300; max-width: 560px; line-height: 1.6; }

/* ── KPI cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 40px; }
.kpi-card { background: #111; border: 1px solid #1a1a1a; border-radius: 20px; padding: 28px 28px 26px; position: relative; overflow: hidden; transition: border-color 0.2s; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--accent, #1DB954); }
.kpi-card:hover { border-color: #2a2a2a; }
.kpi-icon { font-size: 26px; margin-bottom: 16px; display: block; line-height: 1; }
.kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #555; margin-bottom: 10px; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 44px; font-weight: 800; color: #fff; line-height: 1; letter-spacing: -2px; }
.kpi-value span { color: var(--accent, #1DB954); font-size: 28px; letter-spacing: -1px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1a1a1a !important; gap: 0 !important; padding: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #555 !important; font-weight: 500 !important; font-size: 14px !important; padding: 14px 28px !important; border-radius: 0 !important; border-bottom: 2px solid transparent !important; transition: all 0.2s !important; }
.stTabs [data-baseweb="tab"]:hover { color: #e8e6df !important; background: transparent !important; }
.stTabs [aria-selected="true"] { background: transparent !important; color: #ffffff !important; font-weight: 700 !important; border-bottom: 2px solid #1DB954 !important; }

/* ── Section headers ── */
.section-header { margin: 8px 0 32px; padding-bottom: 20px; border-bottom: 1px solid #1a1a1a; }
.section-number { font-size: 11px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: #1DB954; margin-bottom: 8px; }
.section-title { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: #fff; letter-spacing: -0.5px; line-height: 1.2; margin: 0; }
.section-desc { font-size: 14px; color: #555; margin-top: 8px; font-weight: 300; }

/* ── Insight boxes ── */
.insight { background: #111; border: 1px solid #1a1a1a; border-left: 3px solid #1DB954; border-radius: 0 12px 12px 0; padding: 18px 22px; margin: 24px 0 8px; font-size: 14px; line-height: 1.7; color: #888; }
.insight strong { color: #e8e6df; }
.insight .hl { color: #1DB954; font-weight: 600; }
.insight.warn { border-left-color: #E8563A; }
.insight.warn .hl { color: #E8563A; }
.insight.amber { border-left-color: #F59B23; }
.insight.amber .hl { color: #F59B23; }
.insight.purple { border-left-color: #9B72CF; }
.insight.purple .hl { color: #9B72CF; }
.insight.blue { border-left-color: #4FC3F7; }
.insight.blue .hl { color: #4FC3F7; }

/* ── Misc ── */
hr { border-color: #1a1a1a !important; margin: 40px 0 !important; }
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid #1a1a1a !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Paleta ────────────────────────────────────────────────────────────────────
GREEN  = "#1DB954"
CORAL  = "#E8563A"
AMBER  = "#F59B23"
PURPLE = "#9B72CF"
BLUE   = "#4FC3F7"
GRAY   = "#888888"
BG     = "#0a0a0a"
BG2    = "#111111"
BG3    = "#161616"

ERA_COLORS = {
    "Pré-2000":  "#555555",
    "2000s":     "#9B72CF",
    "2010s":     "#4FC3F7",
    "2020–2021": "#1DB954",
    "2022–2023": "#E8563A",
}

GENRE_COLORS = {
    "Pop energético":   "#1DB954",
    "Hip-Hop / Rap":    "#E8563A",
    "Ballad / Acústico":"#4FC3F7",
    "Dance / EDM":      "#F59B23",
    "R&B / Soul":       "#9B72CF",
}

def spotify_layout(fig, title="", height=420, showlegend=False,
                   xaxis_title="", yaxis_title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e8e6df", size=14, family="DM Sans"), x=0.01),
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(family="DM Sans", color=GRAY, size=12),
        xaxis=dict(gridcolor="#1a1a1a", linecolor="#1a1a1a", tickcolor="#2a2a2a",
                   zerolinecolor="#1a1a1a", title_text=xaxis_title,
                   title_font=dict(color="#555", size=11)),
        yaxis=dict(gridcolor="#1a1a1a", linecolor="#1a1a1a", tickcolor="#2a2a2a",
                   zerolinecolor="#1a1a1a", title_text=yaxis_title,
                   title_font=dict(color="#555", size=11)),
        legend=dict(bgcolor=BG2, bordercolor="#1a1a1a", borderwidth=1, font=dict(color=GRAY)),
        margin=dict(l=50, r=30, t=50, b=50),
        hoverlabel=dict(bgcolor="#1a1a1a", bordercolor="#2a2a2a",
                        font=dict(color="#e8e6df", size=12, family="DM Sans")),
        colorway=[GREEN, CORAL, AMBER, PURPLE, BLUE, GRAY],
        height=height, showlegend=showlegend,
    )
    return fig

# ── ETL ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_raw = pd.read_csv("spotify-2023.csv", encoding="utf-8", encoding_errors="replace")
    bad = pd.to_numeric(df_raw["streams"], errors="coerce").isna()
    df  = df_raw[~bad].copy()
    df["streams"] = pd.to_numeric(df["streams"])
    num_cols = [
        "artist_count","released_year","released_month","released_day",
        "in_spotify_playlists","in_spotify_charts","in_apple_playlists",
        "in_apple_charts","in_deezer_playlists","in_deezer_charts","in_shazam_charts",
        "bpm","danceability_%","valence_%","energy_%","acousticness_%",
        "instrumentalness_%","liveness_%","speechiness_%",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["key"] = df["key"].replace("", np.nan).fillna("Unknown")
    df["in_shazam_charts"] = df["in_shazam_charts"].fillna(0)
    df["streams_M"] = df["streams"] / 1e6
    df["era"] = pd.cut(df["released_year"],
        bins=[0,1999,2009,2019,2021,2023],
        labels=["Pré-2000","2000s","2010s","2020–2021","2022–2023"]).astype(str)
    df["collab"] = df["artist_count"].apply(lambda x: "Colaboração" if x > 1 else "Solo")
    return df

@st.cache_data
def cluster_genres(df):
    feats = ["bpm","danceability_%","energy_%","acousticness_%","speechiness_%","valence_%"]
    sub   = df[feats].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(sub)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=feats)
    genre_map = {}
    for i, row in centers.iterrows():
        if row["speechiness_%"] > 15:
            genre_map[i] = "Hip-Hop / Rap"
        elif row["acousticness_%"] > 50:
            genre_map[i] = "Ballad / Acústico"
        elif row["energy_%"] > 70 and row["danceability_%"] > 70:
            genre_map[i] = "Dance / EDM"
        elif row["energy_%"] > 60 and row["valence_%"] > 55:
            genre_map[i] = "Pop energético"
        else:
            genre_map[i] = "R&B / Soul"
    result = df.copy()
    result["genre_cluster"] = np.nan
    result.loc[sub.index, "genre_cluster"] = [genre_map[l] for l in labels]
    return result

df = load_data()
df = cluster_genres(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-text">● Spotify</div>
        <div class="logo-sub">2023 · Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # Module selector
    st.markdown("<div class='mod-selector'>", unsafe_allow_html=True)
    st.markdown("<div class='mod-label'>Módulo</div>", unsafe_allow_html=True)
    modulo = st.radio(
        label="módulo",
        options=["EDA", "Correlações", "Machine Learning"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='padding:0 16px'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#333;font-size:10px;font-weight:600;"
                "letter-spacing:0.1em;text-transform:uppercase;"
                "margin:16px 0 12px'>Filtros</p>", unsafe_allow_html=True)

    eras_all  = ["Pré-2000","2000s","2010s","2020–2021","2022–2023"]
    eras_sel  = st.multiselect("Era de lançamento", eras_all, default=eras_all)
    modes_all = sorted(df["mode"].dropna().unique().tolist())
    modes_sel = st.multiselect("Modo", modes_all, default=modes_all)
    year_min  = int(df["released_year"].min())
    year_max  = int(df["released_year"].max())
    year_range = st.slider("Ano de lançamento", year_min, year_max, (year_min, year_max))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='padding:24px 16px 16px'>"
        "<p style='color:#2a2a2a;font-size:10px;line-height:1.6'>"
        "Dataset: Kaggle<br>Most Streamed Spotify Songs 2023</p>"
        "<p style='color:#2a2a2a;font-size:10px;margin-top:8px'>Desenvolvido por<br>"
        "<a href='https://github.com/SanderAugustoGarcia' "
        "style='color:#1DB954;text-decoration:none'>Sander Augusto Garcia</a></p>"
        "</div>", unsafe_allow_html=True)

# ── Filtro global ─────────────────────────────────────────────────────────────
mask = (
    df["era"].isin(eras_sel) &
    df["mode"].isin(modes_sel) &
    df["released_year"].between(year_range[0], year_range[1])
)
dff = df[mask].copy()

# ── KPIs ──────────────────────────────────────────────────────────────────────
s_arr = dff["streams"].dropna().values
if len(s_arr) > 1:
    s_sorted    = np.sort(s_arr)
    cum_s       = np.cumsum(s_sorted) / s_sorted.sum()
    cum_p       = np.arange(1, len(s_sorted)+1) / len(s_sorted)
    gini        = float(1 - 2 * np.trapezoid(cum_s, cum_p))
    share_top10 = float((1 - cum_s[int(0.9*len(s_sorted))]) * 100)
else:
    s_sorted = cum_s = cum_p = np.array([0])
    gini = share_top10 = 0.0

# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO EDA
# ══════════════════════════════════════════════════════════════════════════════
if modulo == "EDA":

    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">Análise Exploratória de Dados · 2023</div>
        <h1 class="hero-title">Spotify<br><span>em números.</span></h1>
        <p class="hero-sub">Pipeline completo ETL → EDA sobre as
        <strong style="color:#e8e6df">{len(dff):,} músicas</strong>
        mais ouvidas no Spotify em 2023.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="--accent:#1DB954">
        <span class="kpi-icon">🎵</span>
        <div class="kpi-label">Músicas</div>
        <div class="kpi-value">{len(dff):,}</div>
      </div>
      <div class="kpi-card" style="--accent:#4FC3F7">
        <span class="kpi-icon">▶</span>
        <div class="kpi-label">Streams total</div>
        <div class="kpi-value">{dff['streams'].sum()/1e9:.1f}<span>B</span></div>
      </div>
      <div class="kpi-card" style="--accent:#F59B23">
        <span class="kpi-icon">~</span>
        <div class="kpi-label">Mediana streams</div>
        <div class="kpi-value">{dff['streams_M'].median():.0f}<span>M</span></div>
      </div>
      <div class="kpi-card" style="--accent:#9B72CF">
        <span class="kpi-icon">⊘</span>
        <div class="kpi-label">Gini</div>
        <div class="kpi-value">{gini:.2f}</div>
      </div>
      <div class="kpi-card" style="--accent:#E8563A">
        <span class="kpi-icon">↑</span>
        <div class="kpi-label">Top 10% streams</div>
        <div class="kpi-value">{share_top10:.0f}<span>%</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribuição","Sazonalidade","Eras","Colaborações","Top músicas"])

    # ── TAB 1: Distribuição
    with tab1:
        st.markdown("""<div class="section-header">
            <div class="section-number">Q1 · Distribuição</div>
            <h2 class="section-title">Quão desigual é a<br>distribuição de streams?</h2>
            <p class="section-desc">Histograma, escala logarítmica, Curva de Lorenz e Coeficiente de Gini</p>
        </div>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            log_s = np.log10(dff["streams"].clip(lower=1))
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=log_s, nbinsx=40, marker_color=GREEN, opacity=0.9,
                hovertemplate="log₁₀: %{x:.2f}<br>Músicas: %{y}<extra></extra>"))
            fig.add_vline(x=float(np.log10(dff["streams"].median())),
                line_dash="dash", line_color=CORAL, line_width=1.5,
                annotation_text=f"Mediana {dff['streams_M'].median():.0f}M",
                annotation_font_color=CORAL, annotation_font_size=11)
            fig.add_vline(x=float(np.log10(dff["streams"].mean())),
                line_dash="dot", line_color=AMBER, line_width=1.5,
                annotation_text=f"Média {dff['streams_M'].mean():.0f}M",
                annotation_font_color=AMBER, annotation_position="top left",
                annotation_font_size=11)
            spotify_layout(fig, title="Distribuição em log₁₀",
                xaxis_title="log₁₀ (streams)", yaxis_title="Nº de músicas", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=cum_p*100, y=cum_s*100, mode="lines",
                line=dict(color=GREEN, width=2.5), fill="tonexty",
                fillcolor="rgba(29,185,84,0.08)", name="Curva de Lorenz",
                hovertemplate="Top %{x:.1f}% músicas<br>= %{y:.1f}% streams<extra></extra>"))
            fig2.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines",
                line=dict(color="#2a2a2a", width=1.2, dash="dash"), name="Igualdade perfeita"))
            fig2.add_annotation(x=65, y=22,
                text=f"<b>Top 10% = {share_top10:.0f}% streams</b><br>Gini = {gini:.2f}",
                showarrow=True, arrowhead=2, ax=0, ay=-50,
                bgcolor=BG3, bordercolor=GREEN, borderwidth=1,
                font=dict(color=GREEN, size=12, family="DM Sans"))
            spotify_layout(fig2, title="Curva de Lorenz",
                xaxis_title="% músicas (acumulado)", yaxis_title="% streams (acumulado)",
                height=380, showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"""<div class="insight">
            💡 <strong>Insight:</strong> Distribuição em <span class="hl">lei de potência</span> —
            média (<span class="hl">{dff['streams_M'].mean():.0f}M</span>) quase o dobro da mediana
            (<span class="hl">{dff['streams_M'].median():.0f}M</span>).
            Gini de <span class="hl">{gini:.2f}</span>: top 10% gera
            <span class="hl">{share_top10:.0f}%</span> de todos os streams.
        </div>""", unsafe_allow_html=True)

    # ── TAB 2: Sazonalidade
    with tab2:
        st.markdown("""<div class="section-header">
            <div class="section-number">Q2 · Sazonalidade</div>
            <h2 class="section-title">Existe sazonalidade<br>nos lançamentos?</h2>
            <p class="section-desc">Volume de lançamentos e streams mediana por mês — 2022–2023</p>
        </div>""", unsafe_allow_html=True)
        month_names = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
        df_rec  = dff[dff["released_year"] >= 2022]
        m_count = df_rec.groupby("released_month").size().reindex(range(1,13), fill_value=0)
        m_str   = df_rec.groupby("released_month")["streams_M"].median().reindex(range(1,13), fill_value=0)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            colors_vol = [CORAL if v == m_count.max() else GREEN for v in m_count.values]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=month_names, y=m_count.values, marker_color=colors_vol,
                opacity=0.9, text=m_count.values, textposition="outside",
                textfont=dict(color="#e8e6df", size=11),
                hovertemplate="%{x}: <b>%{y} músicas</b><extra></extra>"))
            fig.add_hline(y=m_count.mean(), line_dash="dash", line_color="#2a2a2a")
            spotify_layout(fig, title="Lançamentos por mês", yaxis_title="Nº de músicas", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            colors_str = [AMBER if v == m_str.max() else GREEN for v in m_str.values]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=month_names, y=m_str.values, marker_color=colors_str,
                opacity=0.9, text=[f"{v:.0f}M" for v in m_str.values], textposition="outside",
                textfont=dict(color="#e8e6df", size=11),
                hovertemplate="%{x}: mediana <b>%{y:.0f}M</b><extra></extra>"))
            spotify_layout(fig2, title="Streams mediana por mês", yaxis_title="Streams mediana (M)", height=380)
            st.plotly_chart(fig2, use_container_width=True)
        best_vol = month_names[m_count.idxmax()-1]
        best_str = month_names[m_str.idxmax()-1]
        worst    = month_names[m_count.idxmin()-1]
        st.markdown(f"""<div class="insight amber">
            💡 <strong>Insight:</strong> <span class="hl">{best_vol}</span> lidera em lançamentos
            ({m_count.max()} músicas). <strong>{worst}</strong> é o mais calmo ({m_count.min()}).
            Músicas de <span class="hl">{best_str}</span> têm maior mediana de streams —
            <span class="hl">trade-off entre volume e visibilidade</span>.
        </div>""", unsafe_allow_html=True)

    # ── TAB 3: Eras
    with tab3:
        st.markdown("""<div class="section-header">
            <div class="section-number">Q3 · Eras</div>
            <h2 class="section-title">Músicas antigas ainda<br>dominam os charts?</h2>
            <p class="section-desc">Comparação de streams por era · viés de sobrevivência</p>
        </div>""", unsafe_allow_html=True)
        era_order   = ["Pré-2000","2000s","2010s","2020–2021","2022–2023"]
        era_order_f = [e for e in era_order if e in dff["era"].unique()]
        col1, col2 = st.columns(2, gap="large")
        with col1:
            era_counts = dff["era"].value_counts().reindex(era_order_f, fill_value=0)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=era_counts.index.tolist(), y=era_counts.values,
                marker_color=[ERA_COLORS.get(e, GREEN) for e in era_counts.index], opacity=0.9,
                text=era_counts.values, textposition="outside",
                textfont=dict(color="#e8e6df", size=11),
                hovertemplate="%{x}: <b>%{y} músicas</b><extra></extra>"))
            spotify_layout(fig, title="Músicas por era", yaxis_title="Nº de músicas", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            for era in era_order_f:
                sub   = dff[dff["era"] == era]
                if not len(sub): continue
                color = ERA_COLORS.get(era, GREEN)
                r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                fig2.add_trace(go.Box(y=sub["streams_M"], name=era,
                    marker_color=color, line_color=color,
                    fillcolor=f"rgba({r},{g},{b},0.12)", boxmean=True,
                    hovertemplate=f"<b>{era}</b><br>%{{y:.0f}}M<extra></extra>"))
            spotify_layout(fig2, title="Streams por era (boxplot)", yaxis_title="Streams (M)", height=380)
            st.plotly_chart(fig2, use_container_width=True)
        era_summary = (dff.groupby("era")["streams_M"]
            .agg(["count","median","mean","max"]).reindex(era_order_f)
            .rename(columns={"count":"Músicas","median":"Mediana (M)","mean":"Média (M)","max":"Máx (M)"})
            .round(0).astype({"Músicas": int}))
        st.dataframe(era_summary, use_container_width=True)
        st.markdown("""<div class="insight purple">
            ⚠️ <strong>Viés de sobrevivência:</strong> Os <span class="hl">anos 2000</span>
            têm a maior mediana — mas aqui só estão os <span class="hl">maiores clássicos de sempre</span>.
            Catálogo histórico = <span class="hl">activo passivo de alto valor</span> para editoras.
        </div>""", unsafe_allow_html=True)

    # ── TAB 4: Colaborações
    with tab4:
        st.markdown("""<div class="section-header">
            <div class="section-number">Q4 · Colaborações</div>
            <h2 class="section-title">Colaborações geram mais<br>streams do que solos?</h2>
            <p class="section-desc">Solo vs. Colaboração · análise por número de artistas</p>
        </div>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            counts = dff["collab"].value_counts().reindex(["Solo","Colaboração"], fill_value=0)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=counts.index.tolist(), y=counts.values,
                marker_color=[GREEN, CORAL], opacity=0.9,
                text=[f"{v} ({v/max(len(dff),1)*100:.0f}%)" for v in counts.values],
                textposition="outside", textfont=dict(color="#e8e6df", size=12),
                hovertemplate="%{x}: <b>%{y}</b><extra></extra>"))
            spotify_layout(fig, title="Volume no dataset", yaxis_title="Nº de músicas", height=360)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            medians = dff.groupby("collab")["streams_M"].median().reindex(["Solo","Colaboração"], fill_value=0)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=medians.index.tolist(), y=medians.values,
                marker_color=[GREEN, CORAL], opacity=0.9,
                text=[f"{v:.0f}M" for v in medians.values], textposition="outside",
                textfont=dict(color="#e8e6df", size=14),
                hovertemplate="%{x}: mediana <b>%{y:.0f}M</b><extra></extra>"))
            spotify_layout(fig2, title="Streams mediana", yaxis_title="Streams mediana (M)", height=360)
            st.plotly_chart(fig2, use_container_width=True)
        with col3:
            dff["_ac"] = dff["artist_count"].clip(upper=5)
            by_n = dff.groupby("_ac")["streams_M"].median()
            x_lab = {1:"1 (solo)",2:"2",3:"3",4:"4",5:"5+"}
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=[x_lab.get(int(k), str(k)) for k in by_n.index], y=by_n.values,
                marker_color=[GREEN, BLUE, AMBER, CORAL, PURPLE][:len(by_n)], opacity=0.9,
                text=[f"{v:.0f}M" for v in by_n.values], textposition="outside",
                textfont=dict(color="#e8e6df", size=11),
                hovertemplate="Nº artistas %{x}: <b>%{y:.0f}M</b><extra></extra>"))
            spotify_layout(fig3, title="Mediana por nº de artistas",
                xaxis_title="Nº de artistas", yaxis_title="Streams mediana (M)", height=360)
            st.plotly_chart(fig3, use_container_width=True)
        solo_m   = dff[dff["collab"]=="Solo"]["streams_M"].median()
        collab_m = dff[dff["collab"]=="Colaboração"]["streams_M"].median()
        diff_pct = (solo_m/collab_m - 1)*100 if collab_m > 0 else 0
        st.markdown(f"""<div class="insight warn">
            💡 <strong>Resultado contra-intuitivo:</strong> Solos têm mediana
            <span class="hl">{diff_pct:.0f}% superior</span> ({solo_m:.0f}M vs {collab_m:.0f}M).
            <strong>O factor determinante é a base de fãs do artista principal</strong>,
            não o número de artistas.
        </div>""", unsafe_allow_html=True)

    # ── TAB 5: Top músicas
    with tab5:
        st.markdown("""<div class="section-header">
            <div class="section-number">Top · Ranking</div>
            <h2 class="section-title">As músicas<br>mais ouvidas.</h2>
            <p class="section-desc">Ranking dinâmico — filtra e ordena por qualquer métrica</p>
        </div>""", unsafe_allow_html=True)
        col_ctrl1, col_ctrl2, _ = st.columns([1,1,2])
        with col_ctrl1:
            n_top = st.slider("Nº de músicas", 5, 50, 15)
        with col_ctrl2:
            sort_by = st.selectbox("Ordenar por", [
                "streams","in_spotify_playlists","in_spotify_charts",
                "danceability_%","energy_%","bpm"])
        top15 = dff.nlargest(15, "streams_M")
        fig   = go.Figure()
        fig.add_trace(go.Bar(
            x=top15["streams_M"],
            y=top15["track_name"] + "  ·  " + top15["artist(s)_name"],
            orientation="h",
            marker=dict(color=top15["streams_M"],
                        colorscale=[[0,"#0f5c28"],[1,GREEN]], showscale=False),
            opacity=0.95,
            text=[f"{v:.0f}M" for v in top15["streams_M"]], textposition="outside",
            textfont=dict(color=GREEN, size=11),
            hovertemplate="<b>%{y}</b><br>%{x:.0f}M streams<extra></extra>"))
        spotify_layout(fig, title="Top 15 — streams totais", xaxis_title="Streams (M)", height=520)
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=320, r=100, t=50, b=50))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        top_df = (dff.nlargest(n_top, sort_by)
            [["track_name","artist(s)_name","released_year","streams_M",
              "in_spotify_playlists","bpm","danceability_%","energy_%","mode","key","collab","era"]]
            .rename(columns={"track_name":"Música","artist(s)_name":"Artista(s)",
                "released_year":"Ano","streams_M":"Streams (M)","in_spotify_playlists":"Playlists",
                "bpm":"BPM","danceability_%":"Dance","energy_%":"Energy",
                "mode":"Modo","key":"Key","collab":"Tipo","era":"Era"})
            .reset_index(drop=True))
        top_df.index += 1
        top_df["Streams (M)"] = top_df["Streams (M)"].round(0).astype(int)
        st.dataframe(top_df, use_container_width=True, height=480)

# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO CORRELAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
elif modulo == "Correlações":

    MUSIC_ATTRS = ["bpm","danceability_%","valence_%","energy_%",
                   "acousticness_%","instrumentalness_%","liveness_%","speechiness_%"]
    ATTRS_LABEL = {"bpm":"BPM","danceability_%":"Danceability","valence_%":"Valence",
                   "energy_%":"Energy","acousticness_%":"Acousticness",
                   "instrumentalness_%":"Instrumentalness","liveness_%":"Liveness",
                   "speechiness_%":"Speechiness"}

    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">Correlações · Atributos musicais</div>
        <h1 class="hero-title">O que faz<br><span>uma música?</span></h1>
        <p class="hero-sub">Relações entre BPM, energy, danceability, valence e streams —
        <strong style="color:#e8e6df">{len(dff):,} músicas</strong> analisadas.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Heatmap atributos",
        "Scatter interactivo",
        "Playlists → streams",
        "Major vs. Minor",
        "Por ano",
        "Por género",
        "Cross-platform",
    ])

    # ── C1: Heatmap de correlação
    with tab1:
        st.markdown("""<div class="section-header">
            <div class="section-number">C1 · Heatmap</div>
            <h2 class="section-title">Correlação entre<br>atributos musicais</h2>
            <p class="section-desc">Matriz de correlação de Pearson entre atributos e streams</p>
        </div>""", unsafe_allow_html=True)

        corr_cols = MUSIC_ATTRS + ["streams_M","in_spotify_playlists"]
        corr_labels = [ATTRS_LABEL.get(c, c.replace("_M"," (M)").replace("in_spotify_playlists","Playlists")) for c in corr_cols]
        corr_matrix = dff[corr_cols].corr().round(2)

        fig = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_labels, y=corr_labels,
            colorscale=[[0, CORAL],[0.5, "#1a1a1a"],[1, GREEN]],
            zmid=0, zmin=-1, zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=11, color="#e8e6df"),
            hovertemplate="%{x} × %{y}<br>r = %{z:.2f}<extra></extra>",
        ))
        spotify_layout(fig, title="Matriz de correlação (Pearson)", height=520)
        fig.update_layout(margin=dict(l=120, r=30, t=50, b=120))
        fig.update_xaxes(tickangle=-40, tickfont=dict(size=11))
        fig.update_yaxes(tickfont=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)

        top_corr = corr_matrix["streams_M"].drop("streams_M").abs().sort_values(ascending=False)
        best = top_corr.index[0]
        best_r = corr_matrix["streams_M"][best]
        st.markdown(f"""<div class="insight">
            💡 <strong>Insight:</strong> A correlação mais forte com streams é
            <span class="hl">{ATTRS_LABEL.get(best, best)}</span>
            (r = <span class="hl">{best_r:.2f}</span>).
            No geral as correlações são fracas — os atributos musicais por si só
            <span class="hl">não determinam o sucesso</span>. A promoção e as playlists
            editoriais têm mais peso do que o BPM ou a danceability.
        </div>""", unsafe_allow_html=True)

    # ── C2: Scatter interactivo
    with tab2:
        st.markdown("""<div class="section-header">
            <div class="section-number">C2 · Scatter</div>
            <h2 class="section-title">Atributo vs. streams<br>por era</h2>
            <p class="section-desc">Cada ponto é uma música — selecciona o atributo e a escala</p>
        </div>""", unsafe_allow_html=True)

        col_ctrl1, col_ctrl2 = st.columns([2,1])
        with col_ctrl1:
            attr_x = st.selectbox("Atributo musical (eixo X)",
                options=MUSIC_ATTRS,
                format_func=lambda x: ATTRS_LABEL.get(x, x))
        with col_ctrl2:
            log_scale = st.toggle("Escala log nos streams", value=True)

        era_order_f = [e for e in ["Pré-2000","2000s","2010s","2020–2021","2022–2023"]
                       if e in dff["era"].unique()]
        fig = go.Figure()
        for era in era_order_f:
            sub = dff[dff["era"] == era].dropna(subset=[attr_x, "streams_M"])
            if not len(sub): continue
            y_vals = np.log10(sub["streams_M"].clip(lower=0.01)) if log_scale else sub["streams_M"]
            fig.add_trace(go.Scatter(
                x=sub[attr_x], y=y_vals,
                mode="markers", name=era,
                marker=dict(color=ERA_COLORS.get(era, GRAY), size=5, opacity=0.55),
                hovertemplate=f"<b>%{{customdata[0]}}</b><br>{ATTRS_LABEL.get(attr_x,attr_x)}: %{{x}}<br>Streams: %{{customdata[1]:.0f}}M<extra>{era}</extra>",
                customdata=np.stack([sub["track_name"], sub["streams_M"]], axis=-1),
            ))

        y_label = f"log₁₀(Streams M)" if log_scale else "Streams (M)"
        spotify_layout(fig, title=f"{ATTRS_LABEL.get(attr_x,attr_x)} vs. Streams",
            xaxis_title=ATTRS_LABEL.get(attr_x, attr_x),
            yaxis_title=y_label, height=480, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        r = dff[[attr_x, "streams_M"]].dropna().corr().iloc[0,1]
        st.markdown(f"""<div class="insight blue">
            💡 Correlação de Pearson entre
            <span class="hl">{ATTRS_LABEL.get(attr_x,attr_x)}</span> e streams:
            <span class="hl">r = {r:.3f}</span>.
            {"Correlação fraca — este atributo explica pouca variância nos streams." if abs(r) < 0.2
             else "Correlação moderada — há uma tendência visível."}
        </div>""", unsafe_allow_html=True)

    # ── C3: Playlists → streams
    with tab3:
        st.markdown("""<div class="section-header">
            <div class="section-number">C3 · Playlists</div>
            <h2 class="section-title">Playlists são o principal<br>driver de streams?</h2>
            <p class="section-desc">Correlação entre presença em playlists e número de streams</p>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            sub = dff[["in_spotify_playlists","streams_M"]].dropna()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub["in_spotify_playlists"],
                y=np.log10(sub["streams_M"].clip(lower=0.01)),
                mode="markers",
                marker=dict(color=GREEN, size=4, opacity=0.4),
                hovertemplate="Playlists: %{x}<br>Streams: 10^%{y:.1f}M<extra></extra>",
            ))
            r_pl = sub.corr().iloc[0,1]
            fig.add_annotation(x=sub["in_spotify_playlists"].quantile(0.95),
                y=np.log10(sub["streams_M"].max()),
                text=f"<b>r = {r_pl:.2f}</b>",
                showarrow=False, bgcolor=BG3, bordercolor=GREEN, borderwidth=1,
                font=dict(color=GREEN, size=13))
            spotify_layout(fig, title="Playlists Spotify vs. Streams",
                xaxis_title="Nº de playlists", yaxis_title="log₁₀ Streams (M)", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            platforms = {
                "Spotify Playlists": "in_spotify_playlists",
                "Spotify Charts":    "in_spotify_charts",
                "Apple Playlists":   "in_apple_playlists",
                "Deezer Playlists":  "in_deezer_playlists",
                "Shazam Charts":     "in_shazam_charts",
            }
            r_vals = {k: dff[[v,"streams_M"]].dropna().corr().iloc[0,1]
                      for k, v in platforms.items()}
            r_sorted = dict(sorted(r_vals.items(), key=lambda x: x[1], reverse=True))
            colors_bar = [GREEN if v >= 0.3 else BLUE if v >= 0.1 else GRAY
                          for v in r_sorted.values()]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=list(r_sorted.values()), y=list(r_sorted.keys()),
                orientation="h", marker_color=colors_bar, opacity=0.9,
                text=[f"r = {v:.2f}" for v in r_sorted.values()],
                textposition="outside", textfont=dict(color="#e8e6df", size=11),
                hovertemplate="%{y}: r = %{x:.3f}<extra></extra>",
            ))
            spotify_layout(fig2, title="Correlação por canal vs. streams",
                xaxis_title="Pearson r", height=380)
            fig2.update_layout(xaxis=dict(range=[0, max(r_sorted.values())*1.3]))
            st.plotly_chart(fig2, use_container_width=True)

        best_platform = max(r_vals, key=r_vals.get)
        st.markdown(f"""<div class="insight">
            💡 <strong>Insight:</strong> <span class="hl">{best_platform}</span> é o canal
            com maior correlação com streams (r = <span class="hl">{r_vals[best_platform]:.2f}</span>).
            A presença em playlists editoriais do Spotify é o
            <span class="hl">principal driver de crescimento orgânico</span> —
            mais do que estar nos charts ou no Shazam.
        </div>""", unsafe_allow_html=True)

    # ── C4: Major vs. Minor
    with tab4:
        st.markdown("""<div class="section-header">
            <div class="section-number">C4 · Modo</div>
            <h2 class="section-title">Major vs. Minor —<br>importa o modo musical?</h2>
            <p class="section-desc">Distribuição de valence, energy e streams por modo</p>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        modes_present = [m for m in ["Major","Minor"] if m in dff["mode"].values]
        mode_colors   = {"Major": GREEN, "Minor": CORAL}

        with col1:
            fig = go.Figure()
            for m in modes_present:
                sub = dff[dff["mode"] == m]["valence_%"].dropna()
                fig.add_trace(go.Violin(y=sub, name=m,
                    line_color=mode_colors[m], fillcolor=f"rgba({','.join(str(int(mode_colors[m][i:i+2],16)) for i in (1,3,5))},0.15)",
                    box_visible=True, meanline_visible=True,
                    hovertemplate=f"{m}<br>Valence: %{{y}}<extra></extra>"))
            spotify_layout(fig, title="Distribuição de Valence (positividade)",
                yaxis_title="Valence (%)", height=380, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            mode_stats = dff.groupby("mode")["streams_M"].agg(["median","mean","count"]).round(1)
            fig2 = go.Figure()
            for m in modes_present:
                if m not in mode_stats.index: continue
                sub = dff[dff["mode"] == m]["streams_M"]
                fig2.add_trace(go.Box(y=sub, name=m,
                    marker_color=mode_colors[m], line_color=mode_colors[m],
                    fillcolor=f"rgba({','.join(str(int(mode_colors[m][i:i+2],16)) for i in (1,3,5))},0.12)",
                    boxmean=True,
                    hovertemplate=f"{m}<br>Streams: %{{y:.0f}}M<extra></extra>"))
            spotify_layout(fig2, title="Streams por modo musical",
                yaxis_title="Streams (M)", height=380, showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)

        if len(modes_present) == 2:
            major_med = dff[dff["mode"]=="Major"]["streams_M"].median()
            minor_med = dff[dff["mode"]=="Minor"]["streams_M"].median()
            major_val = dff[dff["mode"]=="Major"]["valence_%"].mean()
            minor_val = dff[dff["mode"]=="Minor"]["valence_%"].mean()
            st.markdown(f"""<div class="insight">
                💡 <strong>Insight:</strong> Major tem valence média de
                <span class="hl">{major_val:.0f}%</span> vs.
                <span class="hl">{minor_val:.0f}%</span> em Minor —
                confirmando que maior = mais positivo musicalmente.
                Em streams, a diferença entre Major (<span class="hl">{major_med:.0f}M</span>)
                e Minor (<span class="hl">{minor_med:.0f}M</span>) é
                <span class="hl">{"pequena — o modo não determina o sucesso comercial" if abs(major_med-minor_med) < 30 else "relevante"}</span>.
            </div>""", unsafe_allow_html=True)

    # ── C5: Heatmap por ano
    with tab5:
        st.markdown("""<div class="section-header">
            <div class="section-number">C5 · Temporal</div>
            <h2 class="section-title">Como evoluíram<br>os atributos por ano?</h2>
            <p class="section-desc">Heatmap de médias dos atributos musicais por ano de lançamento</p>
        </div>""", unsafe_allow_html=True)

        years_with_data = sorted(dff["released_year"].dropna().unique())
        years_recent    = [y for y in years_with_data if y >= 2010]
        attrs_show      = ["bpm","danceability_%","energy_%","valence_%","acousticness_%","speechiness_%"]
        attrs_labels    = [ATTRS_LABEL[a] for a in attrs_show]

        heat_df = (dff[dff["released_year"].isin(years_recent)]
            .groupby("released_year")[attrs_show].mean().round(1))

        from sklearn.preprocessing import MinMaxScaler
        scaler_h = MinMaxScaler()
        heat_norm = pd.DataFrame(scaler_h.fit_transform(heat_df),
            index=heat_df.index, columns=heat_df.columns)

        fig = go.Figure(go.Heatmap(
            z=heat_norm.T.values,
            x=[str(int(y)) for y in heat_norm.index],
            y=attrs_labels,
            colorscale=[[0,"#0f3d22"],[0.5,"#1a1a1a"],[1,GREEN]],
            text=heat_df.T.values.round(0),
            texttemplate="%{text}",
            textfont=dict(size=10, color="#e8e6df"),
            hovertemplate="Ano: %{x}<br>%{y}: %{text}<extra></extra>",
        ))
        spotify_layout(fig, title="Atributos médios por ano (normalizado 0-1)", height=420)
        fig.update_layout(margin=dict(l=120, r=30, t=50, b=60))
        fig.update_xaxes(tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

        energy_trend  = heat_df["energy_%"].iloc[-1] - heat_df["energy_%"].iloc[0]
        acoustic_trend = heat_df["acousticness_%"].iloc[-1] - heat_df["acousticness_%"].iloc[0]
        st.markdown(f"""<div class="insight amber">
            💡 <strong>Insight:</strong> De 2010 a 2023, energy
            {"aumentou" if energy_trend > 0 else "diminuiu"}
            {abs(energy_trend):.0f} pontos e acousticness
            {"aumentou" if acoustic_trend > 0 else "diminuiu"}
            {abs(acoustic_trend):.0f} pontos.
            Os gostos musicais evoluem de forma mensurável —
            <span class="hl">os dados confirmam tendências da indústria</span>
            como o crescimento do pop acústico pós-2020.
        </div>""", unsafe_allow_html=True)

    # ── C6: Heatmap por género inferido
    with tab6:
        st.markdown("""<div class="section-header">
            <div class="section-number">C6 · Géneros inferidos</div>
            <h2 class="section-title">Perfis de géneros<br>sem usar labels</h2>
            <p class="section-desc">K-Means (k=5) sobre atributos musicais — géneros emergentes</p>
        </div>""", unsafe_allow_html=True)

        dff_g = dff.dropna(subset=["genre_cluster"])
        col1, col2 = st.columns(2, gap="large")

        with col1:
            genre_list = [g for g in GENRE_COLORS if g in dff_g["genre_cluster"].unique()]
            heat_genre = (dff_g.groupby("genre_cluster")[
                ["bpm","danceability_%","energy_%","acousticness_%","speechiness_%","valence_%"]]
                .mean().round(0))
            heat_genre_labels = [ATTRS_LABEL[c] for c in heat_genre.columns]

            from sklearn.preprocessing import MinMaxScaler
            sc2 = MinMaxScaler()
            heat_gn = pd.DataFrame(sc2.fit_transform(heat_genre),
                index=heat_genre.index, columns=heat_genre.columns)

            fig = go.Figure(go.Heatmap(
                z=heat_gn.values,
                x=heat_genre_labels,
                y=heat_gn.index.tolist(),
                colorscale=[[0,"#0f3d22"],[0.5,"#1a1a1a"],[1,GREEN]],
                text=heat_genre.values,
                texttemplate="%{text}",
                textfont=dict(size=11, color="#e8e6df"),
                hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
            ))
            spotify_layout(fig, title="Perfil de atributos por género inferido", height=380)
            fig.update_layout(margin=dict(l=150, r=30, t=50, b=80))
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            genre_streams = (dff_g.groupby("genre_cluster")["streams_M"]
                .agg(["median","count"]).sort_values("median", ascending=False))
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=genre_streams.index.tolist(),
                y=genre_streams["median"].values,
                marker_color=[GENRE_COLORS.get(g, GRAY) for g in genre_streams.index],
                opacity=0.9,
                text=[f"{v:.0f}M\n(n={n})" for v,n in
                      zip(genre_streams["median"], genre_streams["count"])],
                textposition="outside", textfont=dict(color="#e8e6df", size=11),
                hovertemplate="%{x}<br>Mediana: %{y:.0f}M<extra></extra>",
            ))
            spotify_layout(fig2, title="Streams mediana por género inferido",
                yaxis_title="Streams mediana (M)", height=380)
            fig2.update_xaxes(tickangle=-15)
            st.plotly_chart(fig2, use_container_width=True)

        best_genre = genre_streams["median"].idxmax()
        st.markdown(f"""<div class="insight purple">
            💡 <strong>Insight (ML aplicado):</strong> Sem usar nenhuma label de género,
            o K-Means identificou <span class="hl">5 perfis distintos</span>
            reconhecíveis musicalmente. O género
            <span class="hl">{best_genre}</span> tem a maior mediana de streams
            ({genre_streams.loc[best_genre,'median']:.0f}M).
            Este clustering é a ponte entre EDA e Machine Learning —
            features não-supervisionadas com valor de negócio real.
        </div>""", unsafe_allow_html=True)

    # ── C7: Cross-platform
    with tab7:
        st.markdown("""<div class="section-header">
            <div class="section-number">C7 · Cross-platform</div>
            <h2 class="section-title">Um hit no Spotify<br>é hit em todo o lado?</h2>
            <p class="section-desc">Consistência de rankings entre Spotify, Apple Music, Deezer e Shazam</p>
        </div>""", unsafe_allow_html=True)

        platform_cols = {
            "Spotify Playlists": "in_spotify_playlists",
            "Spotify Charts":    "in_spotify_charts",
            "Apple Playlists":   "in_apple_playlists",
            "Apple Charts":      "in_apple_charts",
            "Deezer Playlists":  "in_deezer_playlists",
            "Deezer Charts":     "in_deezer_charts",
            "Shazam Charts":     "in_shazam_charts",
        }
        plat_df = dff[list(platform_cols.values())].rename(columns={v:k for k,v in platform_cols.items()})
        plat_corr = plat_df.corr().round(2)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig = go.Figure(go.Heatmap(
                z=plat_corr.values,
                x=plat_corr.columns.tolist(),
                y=plat_corr.index.tolist(),
                colorscale=[[0,"#1a1a1a"],[1,GREEN]],
                zmid=0.5, zmin=0, zmax=1,
                text=plat_corr.values.round(2),
                texttemplate="%{text}",
                textfont=dict(size=11, color="#e8e6df"),
                hovertemplate="%{x} × %{y}<br>r = %{z:.2f}<extra></extra>",
            ))
            spotify_layout(fig, title="Correlação entre plataformas", height=420)
            fig.update_layout(margin=dict(l=130, r=30, t=50, b=130))
            fig.update_xaxes(tickangle=-40, tickfont=dict(size=10))
            fig.update_yaxes(tickfont=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sp_pl  = dff[["in_spotify_playlists","in_apple_playlists"]].dropna()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sp_pl["in_spotify_playlists"],
                y=sp_pl["in_apple_playlists"],
                mode="markers",
                marker=dict(color=GREEN, size=4, opacity=0.4),
                hovertemplate="Spotify: %{x}<br>Apple: %{y}<extra></extra>",
            ))
            r_cross = sp_pl.corr().iloc[0,1]
            fig2.add_annotation(x=sp_pl["in_spotify_playlists"].max()*0.7,
                y=sp_pl["in_apple_playlists"].max()*0.9,
                text=f"<b>r = {r_cross:.2f}</b>",
                showarrow=False, bgcolor=BG3, bordercolor=GREEN, borderwidth=1,
                font=dict(color=GREEN, size=13))
            spotify_layout(fig2, title="Spotify Playlists vs. Apple Playlists",
                xaxis_title="Spotify Playlists", yaxis_title="Apple Playlists", height=420)
            st.plotly_chart(fig2, use_container_width=True)

        best_cross = plat_corr.unstack().drop_duplicates()
        best_cross = best_cross[best_cross < 1].idxmax()
        st.markdown(f"""<div class="insight blue">
            💡 <strong>Insight:</strong> As plataformas têm correlação moderada entre si —
            um hit no Spotify tende a ser hit também nas outras plataformas,
            mas com excepções relevantes.
            <span class="hl">Shazam Charts</span> é o canal com menor correlação
            com os restantes — capta tendências emergentes antes das playlists editoriais.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
else:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.metrics import (r2_score, mean_squared_error,
                                 accuracy_score, classification_report,
                                 confusion_matrix)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import umap.umap_ as umap_lib

    ML_FEATS = [
        "bpm","danceability_%","valence_%","energy_%",
        "acousticness_%","instrumentalness_%","liveness_%","speechiness_%",
        "in_spotify_playlists","artist_count","released_year","released_month",
    ]
    FEAT_LABELS = {
        "bpm":"BPM","danceability_%":"Danceability","valence_%":"Valence",
        "energy_%":"Energy","acousticness_%":"Acousticness",
        "instrumentalness_%":"Instrumentalness","liveness_%":"Liveness",
        "speechiness_%":"Speechiness","in_spotify_playlists":"Playlists Spotify",
        "artist_count":"Nº artistas","released_year":"Ano","released_month":"Mês",
    }

    @st.cache_data
    def prepare_ml(df, feats):
        sub = df[feats + ["streams_M"]].dropna()
        X = sub[feats]
        y_reg = sub["streams_M"]
        threshold = y_reg.quantile(0.75)
        y_clf = (y_reg >= threshold).astype(int)
        return X, y_reg, y_clf, threshold

    @st.cache_data
    def run_regression(X_vals, y_vals, n_est, max_d, seed=42):
        X = pd.DataFrame(X_vals)
        y = pd.Series(y_vals)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                      random_state=seed, n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        r2   = r2_score(y_te, y_pred)
        rmse = mean_squared_error(y_te, y_pred) ** 0.5
        fi   = pd.Series(model.feature_importances_, index=X.columns)
        return r2, rmse, fi, y_te.values, y_pred

    @st.cache_data
    def run_classification(X_vals, y_vals, n_est, max_d, seed=42):
        X = pd.DataFrame(X_vals)
        y = pd.Series(y_vals)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                       random_state=seed, n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        cm  = confusion_matrix(y_te, y_pred)
        fi  = pd.Series(model.feature_importances_, index=X.columns)
        return acc, cm, fi, y_te.values, y_pred

    @st.cache_data
    def run_learning_curve(X_vals, y_vals, n_est, max_d, seed=42):
        X = pd.DataFrame(X_vals)
        y = pd.Series(y_vals)
        model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                      random_state=seed, n_jobs=-1)
        sizes = np.linspace(0.1, 1.0, 8)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=4, train_sizes=sizes,
            scoring="r2", n_jobs=-1)
        return train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)

    @st.cache_data
    def run_umap(X_vals, labels, seed=42):
        X = pd.DataFrame(X_vals)
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        reducer = umap_lib.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        emb = reducer.fit_transform(Xs)
        return emb[:, 0], emb[:, 1]

    X_all, y_reg, y_clf, threshold = prepare_ml(dff, ML_FEATS)

    # ── Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">Machine Learning · Spotify 2023</div>
        <h1 class="hero-title">Prever<br><span>o sucesso.</span></h1>
        <p class="hero-sub">Modelos interactivos treinados sobre
        <strong style="color:#e8e6df">{len(X_all):,} músicas</strong> —
        ajusta os parâmetros e vê o impacto em tempo real.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Parâmetros globais na sidebar (já está dentro do with st.sidebar acima,
    # mas os widgets de ML ficam no corpo principal para maior visibilidade)
    st.markdown("""
    <div style="background:#111;border:1px solid #1a1a1a;border-radius:16px;
                padding:24px 28px;margin-bottom:32px">
        <div style="font-size:11px;font-weight:600;letter-spacing:0.12em;
                    text-transform:uppercase;color:#555;margin-bottom:16px">
            ⚙️ Parâmetros do modelo (aplicam-se a todos os tabs)
        </div>
    """, unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        n_est  = st.slider("Nº de árvores (n_estimators)", 50, 500, 200, step=50,
                           help="Mais árvores = modelo mais estável mas mais lento")
    with pc2:
        max_d  = st.slider("Profundidade máxima (max_depth)", 2, 20, 8,
                           help="Mais profundidade = mais complexo, risco de overfitting")
    with pc3:
        st.markdown(f"""
        <div style="padding:8px 0">
            <div style="font-size:10px;color:#555;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:6px">Features usadas</div>
            <div style="font-size:22px;font-weight:800;color:#fff;
                        font-family:\'Syne\',sans-serif">{len(ML_FEATS)}</div>
            <div style="font-size:11px;color:#444;margin-top:4px">
                atributos musicais + playlists + ano/mês</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    tab_ml1, tab_ml2, tab_ml3, tab_ml4, tab_ml5 = st.tabs([
        "Regressão",
        "Importância de features",
        "Classificação de sucesso",
        "Curva de aprendizagem",
        "UMAP 2D",
    ])

    # ══ ML1: Regressão ═══════════════════════════════════════════════════════
    with tab_ml1:
        st.markdown("""<div class="section-header">
            <div class="section-number">ML1 · Regressão</div>
            <h2 class="section-title">Consegues prever<br>o número de streams?</h2>
            <p class="section-desc">Random Forest Regressor — treino/teste 80/20 · target: streams (M)</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="insight blue" style="margin-bottom:24px">
            📖 <strong>O que é R²?</strong> O coeficiente de determinação mede
            <span class="hl">quanto da variância dos streams o modelo consegue explicar</span>.
            R²=1.0 seria perfeição; R²=0 significa que o modelo não é melhor do que prever
            sempre a média. Em dados de música, R²>0.35 já é um resultado interessante.
        </div>""", unsafe_allow_html=True)

        with st.spinner("A treinar o modelo..."):
            r2, rmse, fi_reg, y_te_r, y_pred_r = run_regression(
                X_all.values, y_reg.values, n_est, max_d)

        kc1, kc2, kc3 = st.columns(3)
        r2_color   = "#1DB954" if r2 > 0.3 else "#F59B23" if r2 > 0.1 else "#E8563A"
        rmse_color = "#4FC3F7"
        kc1.markdown(f"""
        <div class="kpi-card" style="--accent:{r2_color}">
            <div class="kpi-label">R² (teste)</div>
            <div class="kpi-value" style="color:{r2_color}">{r2:.3f}</div>
            <div style="font-size:11px;color:#555;margin-top:8px">
                {"✓ Bom resultado" if r2>0.3 else "⚠ Moderado" if r2>0.1 else "✗ Fraco"}</div>
        </div>""", unsafe_allow_html=True)
        kc2.markdown(f"""
        <div class="kpi-card" style="--accent:{rmse_color}">
            <div class="kpi-label">RMSE</div>
            <div class="kpi-value" style="font-size:36px">{rmse:.0f}<span>M</span></div>
            <div style="font-size:11px;color:#555;margin-top:8px">erro médio em streams</div>
        </div>""", unsafe_allow_html=True)
        kc3.markdown(f"""
        <div class="kpi-card" style="--accent:#9B72CF">
            <div class="kpi-label">Músicas no teste</div>
            <div class="kpi-value" style="font-size:36px">{len(y_te_r)}</div>
            <div style="font-size:11px;color:#555;margin-top:8px">20% do dataset</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style=\'margin-top:24px\'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_te_r, y=y_pred_r, mode="markers",
                marker=dict(color=GREEN, size=5, opacity=0.45),
                hovertemplate="Real: %{x:.0f}M<br>Previsto: %{y:.0f}M<extra></extra>",
                name="Previsões"))
            max_v = max(y_te_r.max(), y_pred_r.max())
            fig.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines",
                line=dict(color=CORAL, width=1.5, dash="dash"), name="Perfeito"))
            spotify_layout(fig, title="Real vs. Previsto (conjunto de teste)",
                xaxis_title="Streams reais (M)", yaxis_title="Streams previstos (M)",
                height=380, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            residuals = y_te_r - y_pred_r
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=y_pred_r, y=residuals, mode="markers",
                marker=dict(color=PURPLE, size=5, opacity=0.45),
                hovertemplate="Previsto: %{x:.0f}M<br>Resíduo: %{y:.0f}M<extra></extra>"))
            fig2.add_hline(y=0, line_dash="dash", line_color=CORAL, line_width=1.5)
            spotify_layout(fig2, title="Resíduos (erro do modelo)",
                xaxis_title="Streams previstos (M)", yaxis_title="Resíduo (M)", height=380)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""<div class="insight">
            💡 <strong>Insight:</strong> R² = <span class="hl">{r2:.3f}</span> significa que o modelo
            explica <span class="hl">{r2*100:.0f}%</span> da variância nos streams.
            O RMSE de <span class="hl">{rmse:.0f}M</span> indica o erro médio das previsões.
            {"Os atributos musicais sozinhos têm poder preditivo limitado — factores externos como marketing e virality não estão capturados." if r2 < 0.4 else "Resultado sólido — as features musicais e de playlists têm poder preditivo real."}
        </div>""", unsafe_allow_html=True)

    # ══ ML2: Importância de features ═════════════════════════════════════════
    with tab_ml2:
        st.markdown("""<div class="section-header">
            <div class="section-number">ML2 · Feature Importance</div>
            <h2 class="section-title">O que o modelo<br>aprendeu?</h2>
            <p class="section-desc">Importância de features do Random Forest — qual variável tem mais peso?</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="insight blue" style="margin-bottom:24px">
            📖 <strong>O que é Feature Importance?</strong> Mede
            <span class="hl">quanto cada variável contribui para reduzir o erro</span>
            nas árvores de decisão. Valores mais altos = variável mais importante para as previsões.
            É uma das formas mais directas de interpretar um modelo de ML.
        </div>""", unsafe_allow_html=True)

        with st.spinner("A calcular importâncias..."):
            _, _, fi_reg, _, _ = run_regression(X_all.values, y_reg.values, n_est, max_d)

        fi_sorted = fi_reg.rename(index=FEAT_LABELS).sort_values(ascending=True)
        colors_fi = [GREEN if v == fi_sorted.max() else
                     BLUE  if v >= fi_sorted.quantile(0.7) else GRAY
                     for v in fi_sorted.values]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fi_sorted.values, y=fi_sorted.index.tolist(),
            orientation="h", marker_color=colors_fi, opacity=0.9,
            text=[f"{v*100:.1f}%" for v in fi_sorted.values],
            textposition="outside", textfont=dict(color="#e8e6df", size=11),
            hovertemplate="%{y}: <b>%{x:.3f}</b><extra></extra>",
        ))
        spotify_layout(fig, title="Importância de features — Regressão (streams)",
            xaxis_title="Importância relativa", height=480)
        fig.update_layout(margin=dict(l=160, r=100, t=50, b=50))
        st.plotly_chart(fig, use_container_width=True)

        top1 = str(fi_sorted.index[-1])
        top2 = str(fi_sorted.index[-2])
        top3 = str(fi_sorted.index[-3])
        playlist_msg = (
            "Se Playlists lidera, confirma o que as correlações já sugeriram "
            "— a distribuição é o factor mais determinante, não os atributos musicais."
            if "Playlist" in top1 or "Playlist" in top2
            else "Os atributos musicais têm mais peso do que as playlists "
                 "— surpreendente e diferente do esperado."
        )
        st.markdown(
            "<div class='insight'>"
            "💡 <strong>Insight:</strong> As 3 features mais importantes são "
            f"<span class='hl'>{top1}</span>, "
            f"<span class='hl'>{top2}</span> e "
            f"<span class='hl'>{top3}</span>. "
            f"{playlist_msg}"
            "</div>",
            unsafe_allow_html=True)

    # ══ ML3: Classificação ═══════════════════════════════════════════════════
    with tab_ml3:
        st.markdown(f"""<div class="section-header">
            <div class="section-number">ML3 · Classificação</div>
            <h2 class="section-title">Hit ou não hit?<br>O modelo decide.</h2>
            <p class="section-desc">Random Forest Classifier — "hit" = top 25% streams (≥ {threshold:.0f}M)</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="insight blue" style="margin-bottom:24px">
            📖 <strong>O que é Accuracy?</strong> A percentagem de músicas
            <span class="hl">correctamente classificadas</span> como hit ou não-hit.
            A matriz de confusão mostra onde o modelo erra — falsos positivos (previu hit,
            não foi) e falsos negativos (não previu hit, foi).
        </div>""", unsafe_allow_html=True)

        with st.spinner("A treinar classificador..."):
            acc, cm, fi_clf, y_te_c, y_pred_c = run_classification(
                X_all.values, y_clf.values, n_est, max_d)

        kc1, kc2, kc3, kc4 = st.columns(4)
        acc_color = "#1DB954" if acc > 0.75 else "#F59B23" if acc > 0.6 else "#E8563A"
        tp = cm[1,1]; fp = cm[0,1]; fn = cm[1,0]; tn = cm[0,0]
        precision = tp/(tp+fp) if (tp+fp) > 0 else 0
        recall    = tp/(tp+fn) if (tp+fn) > 0 else 0

        kc1.markdown(f"""<div class="kpi-card" style="--accent:{acc_color}">
            <div class="kpi-label">Accuracy</div>
            <div class="kpi-value" style="color:{acc_color};font-size:40px">{acc*100:.0f}<span>%</span></div>
        </div>""", unsafe_allow_html=True)
        kc2.markdown(f"""<div class="kpi-card" style="--accent:#4FC3F7">
            <div class="kpi-label">Precision</div>
            <div class="kpi-value" style="font-size:40px">{precision*100:.0f}<span>%</span></div>
            <div style="font-size:11px;color:#555;margin-top:4px">hits correctos previstos</div>
        </div>""", unsafe_allow_html=True)
        kc3.markdown(f"""<div class="kpi-card" style="--accent:#9B72CF">
            <div class="kpi-label">Recall</div>
            <div class="kpi-value" style="font-size:40px">{recall*100:.0f}<span>%</span></div>
            <div style="font-size:11px;color:#555;margin-top:4px">hits reais detectados</div>
        </div>""", unsafe_allow_html=True)
        kc4.markdown(f"""<div class="kpi-card" style="--accent:#F59B23">
            <div class="kpi-label">F1 Score</div>
            <div class="kpi-value" style="font-size:40px">{2*precision*recall/(precision+recall+1e-9)*100:.0f}<span>%</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style=\'margin-top:24px\'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Não-hit (prev.)","Hit (prev.)"],
                y=["Não-hit (real)","Hit (real)"],
                colorscale=[[0,"#0a0a0a"],[1,GREEN]],
                text=cm, texttemplate="<b>%{text}</b>",
                textfont=dict(size=20, color="#e8e6df"),
                hovertemplate="%{y} → %{x}: %{z}<extra></extra>",
            ))
            spotify_layout(fig, title="Matriz de confusão", height=380)
            fig.update_layout(margin=dict(l=120, r=30, t=50, b=80))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fi_clf_s = fi_clf.rename(index=FEAT_LABELS).sort_values(ascending=True)
            colors_fc = [CORAL if v == fi_clf_s.max() else
                         AMBER if v >= fi_clf_s.quantile(0.7) else GRAY
                         for v in fi_clf_s.values]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=fi_clf_s.values, y=fi_clf_s.index.tolist(),
                orientation="h", marker_color=colors_fc, opacity=0.9,
                text=[f"{v*100:.1f}%" for v in fi_clf_s.values],
                textposition="outside", textfont=dict(color="#e8e6df", size=11),
                hovertemplate="%{y}: <b>%{x:.3f}</b><extra></extra>"))
            spotify_layout(fig2, title="Features mais importantes (classificação)",
                xaxis_title="Importância", height=380)
            fig2.update_layout(margin=dict(l=160, r=80, t=50, b=50))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""<div class="insight warn">
            💡 <strong>Insight:</strong> O modelo classifica hits com
            <span class="hl">{acc*100:.0f}%</span> de accuracy.
            Precision de <span class="hl">{precision*100:.0f}%</span> significa que
            quando prevê um hit, acerta <span class="hl">{precision*100:.0f}%</span> das vezes.
            Recall de <span class="hl">{recall*100:.0f}%</span> indica que detecta
            <span class="hl">{recall*100:.0f}%</span> dos hits reais.
        </div>""", unsafe_allow_html=True)

    # ══ ML4: Curva de aprendizagem ════════════════════════════════════════════
    with tab_ml4:
        st.markdown("""<div class="section-header">
            <div class="section-number">ML4 · Curva de aprendizagem</div>
            <h2 class="section-title">O modelo tem<br>overfitting?</h2>
            <p class="section-desc">Train vs. Validation score em função do tamanho do dataset</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="insight blue" style="margin-bottom:24px">
            📖 <strong>Overfitting vs. Underfitting:</strong>
            Se o score de treino é muito superior ao de validação →
            <span class="hl">overfitting</span> (modelo decorou os dados).
            Se ambos são baixos → <span class="hl">underfitting</span> (modelo simples demais).
            O ideal é que as duas curvas convirjam num valor alto.
        </div>""", unsafe_allow_html=True)

        with st.spinner("A calcular curvas de aprendizagem (pode demorar alguns segundos)..."):
            tr_sizes, tr_scores, val_scores = run_learning_curve(
                X_all.values, y_reg.values, n_est, max_d)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tr_sizes, y=tr_scores, mode="lines+markers",
            name="Treino", line=dict(color=GREEN, width=2.5),
            marker=dict(size=8, color=GREEN),
            hovertemplate="n=%{x}<br>R² treino: %{y:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=tr_sizes, y=val_scores, mode="lines+markers",
            name="Validação (CV)", line=dict(color=CORAL, width=2.5, dash="dash"),
            marker=dict(size=8, color=CORAL),
            hovertemplate="n=%{x}<br>R² validação: %{y:.3f}<extra></extra>"))
        fig.add_hrect(y0=max(val_scores)-0.02, y1=max(val_scores)+0.02,
            fillcolor="rgba(29,185,84,0.05)", line_width=0)
        spotify_layout(fig, title="Curva de aprendizagem — R² em função do tamanho de treino",
            xaxis_title="Nº de amostras de treino", yaxis_title="R²",
            height=440, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        gap = tr_scores[-1] - val_scores[-1]
        st.markdown(f"""<div class="insight {"warn" if gap > 0.2 else ""}">
            💡 <strong>Diagnóstico:</strong>
            Gap treino/validação = <span class="hl">{gap:.3f}</span>.
            {"⚠️ Gap elevado — sinais de overfitting. Reduz max_depth ou aumenta os dados." if gap > 0.2
             else "✓ Gap controlado — o modelo generaliza bem para dados não vistos."}
            R² de validação estabiliza em
            <span class="hl">{val_scores[-1]:.3f}</span> com o dataset completo.
        </div>""", unsafe_allow_html=True)

    # ══ ML5: UMAP 2D ══════════════════════════════════════════════════════════
    with tab_ml5:
        st.markdown("""<div class="section-header">
            <div class="section-number">ML5 · UMAP</div>
            <h2 class="section-title">O espaço das músicas<br>em 2 dimensões.</h2>
            <p class="section-desc">UMAP reduz 12 dimensões a 2 — cada ponto é uma música</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="insight blue" style="margin-bottom:24px">
            📖 <strong>O que é UMAP?</strong>
            <span class="hl">Uniform Manifold Approximation and Projection</span> —
            um algoritmo que comprime múltiplas dimensões num plano 2D
            <span class="hl">preservando a estrutura de vizinhança</span>.
            Músicas parecidas ficam próximas; músicas diferentes ficam afastadas.
            É das visualizações mais poderosas em DS.
        </div>""", unsafe_allow_html=True)

        color_by = st.selectbox("Colorir pontos por",
            ["Era", "Género inferido", "Streams (M)", "Modo (Major/Minor)"])

        with st.spinner("A projectar com UMAP (pode demorar ~10s)..."):
            sub_umap = dff[ML_FEATS + ["streams_M","era","genre_cluster","mode"]].dropna()
            x_emb, y_emb = run_umap(sub_umap[ML_FEATS].values,
                                    sub_umap["era"].values)

        fig = go.Figure()
        if color_by == "Era":
            for era in ["Pré-2000","2000s","2010s","2020–2021","2022–2023"]:
                mask_e = sub_umap["era"] == era
                if not mask_e.any(): continue
                fig.add_trace(go.Scatter(
                    x=x_emb[mask_e], y=y_emb[mask_e], mode="markers",
                    name=era, marker=dict(color=ERA_COLORS.get(era,GRAY), size=4, opacity=0.55),
                    hovertemplate=f"<b>{era}</b><br>%{{customdata}}<extra></extra>",
                    customdata=sub_umap.loc[mask_e, "streams_M"].round(0).astype(str) + "M"))
            showleg = True

        elif color_by == "Género inferido":
            for g, c in GENRE_COLORS.items():
                mask_g = sub_umap["genre_cluster"] == g
                if not mask_g.any(): continue
                fig.add_trace(go.Scatter(
                    x=x_emb[mask_g], y=y_emb[mask_g], mode="markers",
                    name=g, marker=dict(color=c, size=4, opacity=0.55),
                    hovertemplate=f"<b>{g}</b><br>%{{customdata}}<extra></extra>",
                    customdata=sub_umap.loc[mask_g,"streams_M"].round(0).astype(str)+"M"))
            showleg = True

        elif color_by == "Streams (M)":
            fig.add_trace(go.Scatter(
                x=x_emb, y=y_emb, mode="markers",
                marker=dict(color=np.log10(sub_umap["streams_M"].clip(lower=0.01)),
                            colorscale=[[0,"#0f3d22"],[1,GREEN]],
                            size=4, opacity=0.6, showscale=True,
                            colorbar=dict(title="log₁₀(M)", tickfont=dict(color=GRAY))),
                hovertemplate="Streams: %{customdata:.0f}M<extra></extra>",
                customdata=sub_umap["streams_M"]))
            showleg = False

        else:  # Modo
            for m, c in [("Major", GREEN), ("Minor", CORAL)]:
                mask_m = sub_umap["mode"] == m
                if not mask_m.any(): continue
                fig.add_trace(go.Scatter(
                    x=x_emb[mask_m], y=y_emb[mask_m], mode="markers",
                    name=m, marker=dict(color=c, size=4, opacity=0.55),
                    hovertemplate=f"<b>{m}</b><extra></extra>"))
            showleg = True

        spotify_layout(fig, title=f"UMAP 2D — colorido por {color_by}",
            xaxis_title="UMAP 1", yaxis_title="UMAP 2",
            height=520, showlegend=showleg)
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""<div class="insight purple">
            💡 <strong>Insight:</strong> O UMAP revela a
            <span class="hl">estrutura latente do espaço musical</span>.
            Clusters visíveis confirmam que os atributos musicais têm poder
            discriminante — músicas parecidas agrupam-se naturalmente.
            Muda a coloração para explorar diferentes perspectivas dos mesmos dados.
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:8px 0 24px;color:#2a2a2a;font-size:12px">
    <span>🎵 Spotify 2023 · Dataset: Kaggle</span>
    <span>Desenvolvido por
        <a href="https://github.com/SanderAugustoGarcia"
           style="color:#1DB954;text-decoration:none">Sander Augusto Garcia</a>
    </span>
</div>
""", unsafe_allow_html=True)
