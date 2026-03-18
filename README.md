# 🎵 Spotify 2023 — ETL & Análise Exploratória de Dados

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18-3F4F75?style=flat&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completo-1DB954?style=flat)

> Desenvolvido por **[Sander Augusto Garcia](https://github.com/SanderAugustoGarcia)**

---

## 🚀 Live Demo

> **👉 [Abrir Dashboard Interactivo](https://edaspotify.streamlit.app)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_dark.svg)](https://edaspotify.streamlit.app)

---

## 📌 Sobre o projecto

Análise exploratória das **952 músicas mais ouvidas no Spotify em 2023**, combinando atributos musicais técnicos (BPM, danceability, energy…) com métricas reais de negócio em 4 plataformas: **Spotify, Apple Music, Deezer e Shazam**.

O projecto cobre o pipeline completo de um projecto de dados real:

```
Raw CSV  →  ETL (limpeza + features)  →  EDA (4 perguntas de negócio)  →  Dashboard interactivo
```

> **Dataset:** [Most Streamed Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023) — Kaggle

---

## 📂 Estrutura do repositório

```
spotify-2023-eda/
│
├── app.py                     # Dashboard Streamlit (webapp interactiva)
├── spotify_eda.ipynb          # Notebook EDA completo
├── spotify-2023.csv           # Dataset original
├── requirements.txt           # Dependências Python
│
└── README.md
```

---

## ⚙️ ETL — Problemas encontrados e decisões

| Problema | Coluna | Dimensão | Decisão |
|----------|--------|----------|---------|
| Dados de BPM/key corrompidos na coluna streams | `streams` | 1 linha | Remover linha |
| Valores em branco | `key` | 95 linhas (10%) | Substituir por `'Unknown'` |
| Valores nulos | `in_shazam_charts` | 50 linhas (5%) | Substituir por `0` |

**Features criadas no ETL:**

- `released_date` — data de lançamento parseada
- `era` — período de lançamento (Pré-2000 / 2000s / 2010s / 2020–2021 / 2022–2023)
- `collab` — classificação Solo vs. Colaboração
- `streams_M` — streams em milhões (legibilidade)

---

## 📊 4 Perguntas de Negócio

### Q1 — Quão desigual é a distribuição de streams?

| Métrica | Valor |
|---------|-------|
| Gini | **0.53** |
| Top 10% das músicas | **37%** de todos os streams |
| Média | 514M streams |
| Mediana | 291M streams |

💡 A distribuição segue uma **lei de potência** — cauda longa extrema. O Gini de 0.53 é superior ao da desigualdade de rendimento em Portugal (~0.33). A transformação log₁₀ aproxima os dados de uma normal, relevante para modelação futura.

---

### Q2 — Existe sazonalidade nos lançamentos musicais?

| Observação | Valor |
|------------|-------|
| Mês com mais lançamentos | **Janeiro** (134 músicas) |
| Mês com mais streams/música | **Outubro** |
| Mês mais calmo | **Agosto** (46 músicas) |

💡 Trade-off claro: Janeiro/Maio concentram lançamentos, mas **Outubro** tem a maior mediana de streams por música — lançar com menos concorrência pode valer mais.

---

### Q3 — Músicas antigas ainda dominam os charts de 2023?

| Era | n | Mediana streams |
|-----|---|----------------|
| Pré-2000 | 48 | 622M |
| **2000s** | 20 | **1.229M** ← maior |
| 2010s | 151 | 984M |
| 2020–2021 | 156 | 564M |
| 2022–2023 | 577 | 179M |

💡 **Viés de sobrevivência** — as músicas antigas aqui são apenas os maiores clássicos. O catálogo histórico é um activo de alto valor passivo para editoras.

---

### Q4 — Colaborações geram mais streams do que solos?

| Tipo | Mediana streams |
|------|----------------|
| **Solo** | **334M** |
| Colaboração | 237M |
| Diferença | Solo +41% |

💡 Resultado contra-intuitivo — solos têm mediana 41% superior. O factor determinante é a **base de fãs do artista principal**, não o número de artistas.

---

## 🔑 Sumário Executivo

| # | Pergunta | Resposta-chave | Implicação de negócio |
|---|----------|---------------|----------------------|
| Q1 | Distribuição de streams | Gini 0.53 · top 10% = 37% streams | Mercado concentrado — algoritmo é decisivo |
| Q2 | Sazonalidade | Jan/Mai = mais lançamentos; Out = mais streams/música | Trade-off entre volume e visibilidade |
| Q3 | Músicas antigas | 2000s têm mediana 6× maior que 2022–2023 | Viés de sobrevivência; catálogo = activo passivo |
| Q4 | Colaborações | Solos com mediana 41% superior | Base de fãs do artista principal é o factor decisivo |

---

## 🚀 Como correr localmente

```bash
# 1. Clonar o repositório
git clone https://github.com/SanderAugustoGarcia/spotify-2023-eda.git
cd spotify-2023-eda

# 2. Instalar dependências
pip install -r requirements.txt

# 3a. Abrir o notebook
jupyter notebook spotify_eda.ipynb

# 3b. OU correr a webapp
streamlit run app.py
```

---

## 🛠️ Stack técnica

| Ferramenta | Utilização |
|------------|-----------|
| `pandas` | Carregamento, limpeza e transformação dos dados |
| `numpy` | Cálculos estatísticos (Gini, log, percentis) |
| `plotly` | Visualizações interactivas com hover e filtros |
| `streamlit` | Dashboard web interactivo com deploy gratuito |
| `jupyter` | Notebook narrativo com análise documentada |

---


---

<p align="center">
  Desenvolvido por <strong>Sander Augusto Garcia</strong><br>
  <a href="https://github.com/SanderAugustoGarcia">github.com/SanderAugustoGarcia</a>
</p>
