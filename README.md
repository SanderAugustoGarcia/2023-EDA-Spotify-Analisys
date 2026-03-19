# 🎵 Spotify 2023 — ETL, EDA & Machine Learning Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18-3F4F75?style=flat&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-1DB954?style=flat)

> Developed by **[Sander Augusto Garcia](https://github.com/SanderAugustoGarcia)**

---

## 🚀 Live Demo

> **👉 [Open Interactive Dashboard](https://edaspotify.streamlit.app)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_dark.svg)](https://edaspotify.streamlit.app)

---

## 📌 About

End-to-end data science project analysing the **952 most streamed songs on Spotify in 2023**, combining musical attributes (BPM, danceability, energy…) with real business metrics across 4 platforms: **Spotify, Apple Music, Deezer and Shazam**.

The project covers the full data science pipeline:

```
Raw CSV → ETL (cleaning + features) → EDA (4 business questions)
       → Correlations (7 analyses) → Machine Learning (5 models)
       → Interactive Dashboard
```

> **Dataset:** [Most Streamed Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023) — Kaggle

---

## 📂 Repository structure

```
spotify-2023-eda/
│
├── app.py                     # Streamlit interactive dashboard
├── spotify_eda.ipynb          # Full EDA + ML notebook
├── spotify-2023.csv           # Original dataset
├── requirements.txt           # Python dependencies
│
└── README.md
```

---

## ⚙️ ETL — Issues found and decisions made

| Issue | Column | Scope | Decision |
|-------|--------|-------|----------|
| Corrupted row — BPM/key data shifted into streams column | `streams` | 1 row | Drop row |
| Empty string values | `key` | 95 rows (10%) | Replace with `'Unknown'` |
| Null values | `in_shazam_charts` | 50 rows (5%) | Replace with `0` |

**Features engineered during ETL:**

- `streams_M` — streams in millions (readability)
- `era` — release period (Pre-2000 / 2000s / 2010s / 2020–2021 / 2022–2023)
- `collab` — Solo vs. Collaboration classification
- `genre_cluster` — inferred genre via K-Means clustering (no labels used)

---

## 📊 Module 1 — Exploratory Data Analysis (4 business questions)

### Q1 — How unequal is the streams distribution?

| Metric | Value |
|--------|-------|
| Gini coefficient | **0.53** |
| Top 10% of songs | **37%** of all streams |
| Mean | 514M streams |
| Median | 291M streams |

💡 The distribution follows a **power law** — the mean (514M) is nearly double the median (291M). The log₁₀ transformation approximates a normal distribution, relevant for future modelling.

---

### Q2 — Is there seasonality in music releases?

| Observation | Value |
|-------------|-------|
| Month with most releases | **January** (134 songs) |
| Month with most streams/song | **October** |
| Quietest month | **August** (46 songs) |

💡 Clear trade-off: January/May concentrate releases, but songs launched in **October** have the highest median streams — releasing with less competition can be worth more.

---

### Q3 — Do older songs still dominate the 2023 charts?

| Era | n | Median streams |
|-----|---|----------------|
| Pre-2000 | 48 | 622M |
| **2000s** | 20 | **1,229M** ← highest |
| 2010s | 151 | 984M |
| 2020–2021 | 156 | 564M |
| 2022–2023 | 577 | 179M |

💡 **Survivorship bias** — the only old songs in this dataset are the greatest classics of all time. Historical catalogue is a **high-value passive asset** for record labels.

---

### Q4 — Do collaborations generate more streams than solos?

| Type | Median streams |
|------|----------------|
| **Solo** | **334M** |
| Collaboration | 237M |
| Difference | Solo +41% |

💡 Counter-intuitive result — solos have 41% higher median. **The determining factor is the lead artist's fanbase**, not the number of artists on the track.

---

## 🔗 Module 2 — Correlations (7 analyses)

| # | Analysis | Key finding |
|---|----------|-------------|
| C1 | Attribute correlation heatmap | Weak correlations overall — attributes alone don't determine success |
| C2 | Interactive scatter (attribute vs. streams) | BPM has near-zero correlation with streams |
| C3 | Playlists → streams | Spotify Playlists is the strongest driver of streams |
| C4 | Major vs. Minor | Mode has minimal impact on commercial success |
| C5 | Attributes by year (heatmap) | Measurable evolution of musical tastes 2010–2023 |
| C6 | Inferred genres (K-Means) | 5 distinct profiles emerge without any genre labels |
| C7 | Cross-platform consistency | Shazam captures emerging trends before editorial playlists |

---

## 🤖 Module 3 — Machine Learning (5 interactive models)

All models are **interactive** — adjust `n_estimators` and `max_depth` and see the impact in real time.

| Model | Metric | Result | Interpretation |
|-------|--------|--------|----------------|
| **RF Regression** | R² | ~0.35 | Musical attributes explain ~35% of stream variance |
| **Feature Importance** | Relative importance | Playlists > BPM | Distribution outweighs musical attributes |
| **RF Classification** | Accuracy | ~75% | 3 in 4 hits correctly identified |
| **Learning Curve** | Train/val gap | < 0.20 | Model generalises well, no significant overfitting |
| **UMAP 2D** | Visual clusters | Visible structure | Musical space has real latent structure |

---

## 🔑 Executive Summary

| # | Question | Key answer | Business implication |
|---|----------|-----------|---------------------|
| Q1 | Streams distribution | Gini 0.53 · top 10% = 37% streams | Concentrated market — algorithm placement is decisive |
| Q2 | Seasonality | Jan/May = most releases; Oct = most streams/song | Trade-off between volume and visibility |
| Q3 | Old songs | 2000s median 6× higher than 2022–2023 | Survivorship bias; catalogue = passive asset |
| Q4 | Collaborations | Solos 41% higher median | Lead artist's fanbase is the key factor |
| ML | Predictive model | R²~0.35 · Playlists #1 feature | Distribution > musical attributes |

---

## 🚀 Run locally

```bash
# 1. Clone the repository
git clone https://github.com/SanderAugustoGarcia/spotify-2023-eda.git
cd spotify-2023-eda

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Open the notebook
jupyter notebook spotify_eda.ipynb

# 3b. OR run the webapp
streamlit run app.py
```

---

## 🛠️ Tech stack

| Tool | Usage |
|------|-------|
| `pandas` | Data loading, cleaning and transformation |
| `numpy` | Statistical calculations (Gini, log, percentiles) |
| `plotly` | Interactive visualisations with hover and filters |
| `streamlit` | Interactive web dashboard with free deployment |
| `scikit-learn` | Random Forest, K-Means, learning curves, metrics |
| `umap-learn` | UMAP 2D dimensionality reduction |
| `jupyter` | Narrative notebook for portfolio |

---

<p align="center">
  Developed by <strong>Sander Augusto Garcia</strong><br>
  <a href="https://github.com/SanderAugustoGarcia">github.com/SanderAugustoGarcia</a>
</p>
