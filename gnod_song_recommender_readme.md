# Gnod Song Recommender

A music recommendation project that combines **taste-independent chart browsing** (Billboard Hot 100) with **acoustic similarity matching** (unsupervised clustering on 9 continuous audio features) inside an interactive **Streamlit** app.

**Live App:** [gnod-recommender-sofija-sarafejeva.streamlit.app](https://gnod-recommender-sofija-sarafejeva.streamlit.app/)

## Project Goal

Group songs purely by **how they sound** using unsupervised machine learning—without passing genre labels into the model—and pair that acoustic matching with live Billboard Hot 100 chart recommendations. Genre labels are kept strictly as an external answer key to evaluate cluster quality and recommendation hit rates.

## Repository Structure

```
.
├── gnod_song_recommender.ipynb   # End-to-end pipeline: scraping, APIs, clustering & evaluation
├── app.py                        # Interactive Streamlit recommender app (Chart + Sound Match)
├── billboard_hot100.csv          # Cached Billboard Hot 100 chart (rank, title, artist)
├── gnod_songs.csv                # Cleaned 1,074-track catalog across 16 genres + audio features
├── hot100_songs.csv              # Optional: Billboard Hot 100 tracks enriched with audio features
├── .env                          # Local Spotify API credentials (not committed; add to .gitignore)
└── README.md

```

## Pipeline Overview

### 1. Web Scraping (`billboard_hot100.csv`)

* Scrapes the live **Billboard Hot 100** chart (rank, title, artist) using `requests` and `BeautifulSoup`.

* Saves results locally to avoid redundant requests and powers the taste-independent **Chart** mode.

### 2. API Data Collection (`gnod_songs.csv`)

* **Spotify Web API (`spotipy`):** Paginates `sp.search(q="genre:...")` (`limit=10`, incrementing `offset`) across **16 sonically distinct genres**:
  `rock`, `jazz`, `hip-hop`, `electronic`, `reggae`, `country`, `blues`, `metal`, `pop`, `latin`, `soul`, `funk`, `techno`, `punk`, `folk`, and `flamenco`.

* **ReccoBeats API:** Fetches continuous audio features in batches of up to **40 track IDs** per request and matches results back to Spotify IDs via track `href`.

* **Resumable Caching:** Saves progress to `gnod_songs.csv` after every genre with request delays to stay within Spotify API rate limits. Noisy/low-coverage genres (`classical`, `acoustic`, `k-pop`) were filtered out, leaving **1,074 unique tracks**.

### 3. Feature Selection & Scaling

* Uses **9 continuous audio features**: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, and `tempo`.

* Discrete music-theory tags (`key` and `mode`) were intentionally excluded so sonically similar tracks are not artificially split by major vs. minor key.

* Standardized with `StandardScaler` so high-magnitude features (`tempo`, `loudness`) do not dominate `0–1` bounded features.

## Model Evaluation & Key Findings

### Choosing `k = 6` for K-Means

Sweeping `k = 2..15` across internal clustering metrics showed two levels of structure:

* `k = 3` peaks on raw Silhouette (\~0.252) as three broad mood buckets.

* **`k = 6`** reaches the **global minimum on Davies-Bouldin (\~1.47)** with a local **Silhouette rebound (\~0.178)** and a Calinski-Harabasz plateau (\~186), offering the best balance of distinct musical vibes.

### Model Comparison (`k = 6`, 9 Scaled Features)

| Model | Clusters | Noise (%) | Largest Cluster (%) | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI (vs. Genre) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **K-Means (Winner)** | **6** | **0.0%** | **29.6%** | **0.1775** | **185.98** | **1.4732** | **0.1078** | 
| Agglomerative (Ward) | 6 | 0.0% | 31.0% | 0.1395 | 151.68 | 1.6970 | 0.1029 | 
| Agglomerative (Complete) | 6 | 0.0% | 85.9% | 0.1788 | 65.03 | 1.2506 | 0.0094 | 
| DBSCAN (`eps=1.8`) | 2 | 12.1% | 98.9% | 0.2832 | 25.75 | 0.8774 | 0.0070 | 
| Agglomerative (Average) | 6 | 0.0% | 97.5% | 0.3867 | 23.83 | 0.7920 | 0.0011 | 

* **Why K-Means wins:** Raw Silhouette and Davies-Bouldin are easily fooled by degenerate models (`average`/`complete` linkage and `DBSCAN`) that dump 86%–99% of songs into one mega-cluster or drop tracks as noise. **Calinski-Harabasz**, cluster balance (`largest_%`, `noise_%`), and **ARI** all agree on **K-Means (`k = 6`)**, which covers 100% of the catalog and supports `.predict()` for new songs.

### The 6 Sonic Clusters

* **Cluster 0 — Mellow, Acoustic & Low-Energy:** Jazz, folk, flamenco, soul, blues, country.

* **Cluster 1 — Live-Sounding & High-Energy:** High `liveness` (0.50); metal, electronic, punk, rock.

* **Cluster 2 — Instrumental & Dance-Groovy:** High `instrumentalness` (0.83); cleanly isolates techno alongside funk/electronic.

* **Cluster 3 — Upbeat, Danceable & Feel-Good:** High `danceability` (0.74) and `valence` (0.70); reggae, latin, pop, soul, country.

* **Cluster 4 — Fast, Loud & Driving:** Highest `energy` (0.83), loudest (-5.18 dB), fastest `tempo` (134 BPM); punk, rock, metal.

* **Cluster 5 — Speech-Heavy & Rhythmic:** Standout `speechiness` (0.40); hip-hop and vocal-heavy reggae.

> **Borderline tracks:** Point-level silhouette analysis shows the lowest-scoring tracks (-0.15 to -0.10) belong to hybrid/groove genres—**reggae, hip-hop, blues, funk, and beat-driven folk**—that naturally sit between neighboring vibes.

### Recommender Hit-Rate Benchmark (`n = 5`)

| Method | Same-Genre Hit Rate | Lift vs. Random | 
| ----- | ----- | ----- | 
| `closest_all` (nearest neighbors, full catalog) | **27.0%** | **\~4.7x** | 
| `closest_cluster` (nearest neighbors, same cluster) | **26.3%** | **\~4.5x** | 
| `random_cluster` (random picks, same cluster) | **15.0%** | **\~2.6x** | 
| `random_all` (baseline) | **5.8%** | 1.0x | 

Even though `closest_all` slightly edges out `closest_cluster`, keeping clusters narrows candidate search on larger catalogs, organizes songs into interpretable vibes, and enables variety/discovery via `random_cluster`.

## Getting Started

### 1. Prerequisites & Installation

```
pip install pandas numpy matplotlib scikit-learn requests beautifulsoup4 spotipy python-dotenv streamlit

```

### 2. Configure Spotify Credentials (for Data Collection)

Create a `.env` file in the project root (and ensure `.env` is in `.gitignore`):

```
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret

```

### 3. Run the Pipeline Notebook

Open and run `gnod_song_recommender.ipynb` to scrape the Billboard Hot 100, collect or load `gnod_songs.csv`, evaluate clustering models, and benchmark recommender hit rates.

### 4. Try or Launch the Streamlit App

Try the deployed app directly here: [**https://gnod-recommender-sofija-sarafejeva.streamlit.app/**](https://gnod-recommender-sofija-sarafejeva.streamlit.app/)

To run it locally, make sure `billboard_hot100.csv` and `gnod_songs.csv` are in the same folder as `app.py`, then run:

```
streamlit run app.py

```

### App Highlights

* **Chart View:** Browse the Billboard Hot 100 by nearby ranks or from #1, with automatic cross-matching into the Gnod catalog.

* **Sound Match View:** Search by song or artist, toggle between **Same cluster** and **Whole catalog**, filter by same artist or same genre, inspect a side-by-side 0–1 audio feature bar chart, and listen via embedded Spotify players.

## Tech Stack

* **Data Collection:** Python, `requests`, `BeautifulSoup`, `spotipy`, ReccoBeats API

* **Data & ML:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`

* **App & Deployment:** `streamlit`