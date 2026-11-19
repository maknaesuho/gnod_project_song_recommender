"""Recommend Billboard chart songs or Gnod tracks that sound similar."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from IPython.display import IFrame
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Song recommender", page_icon=":material/music_note:", layout="wide")

DATA_DIR = Path(__file__).resolve().parent
CHART_PATH = DATA_DIR / "billboard_hot100.csv"
CATALOG_PATH = DATA_DIR / "gnod_songs.csv"
N_CLUSTERS = 6

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
UNIT_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]
VIEW_CHART = "Chart"
VIEW_SOUND = "Sound match"
MODE_NEIGHBORS = "neighbors"
MODE_TOP = "top"
METHOD_CLUSTER = "closest_cluster"
METHOD_ALL = "closest_all"

_TOKEN = re.compile(r"[a-z0-9]+")
_ARTIST_STOPWORDS = {"feat", "ft", "featuring", "and", "the", "with", "x"}


@dataclass
class SoundModel:
    songs: pd.DataFrame
    distances: np.ndarray
    track_keys: np.ndarray
    features: list[str]
    silhouette: float
    hit_rates: dict[str, float]
    cluster_sizes: dict[int, int]


def play_song(track_id: str) -> IFrame:
    return IFrame(
        src=f"https://open.spotify.com/embed/track/{track_id}",
        width="320",
        height="80",
        frameborder="0",
        allowtransparency="true",
        allow="encrypted-media",
    )


def _spotify_track_id(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if "/track/" in text:
        text = text.split("/track/", 1)[1].split("?", 1)[0].strip("/")
    return text or None


def _show_player(track_id: str, song: str, artist: str) -> None:
    frame = play_song(track_id)
    st.components.v1.iframe(
        frame.src,
        width=int(frame.width),
        height=int(frame.height),
    )
    st.caption(f"Player: {song} — {artist}")


def _render_players(tracks: list[tuple[str, str, str]]) -> None:
    playable = [(track_id, song, artist) for track_id, song, artist in tracks if track_id]
    if not playable:
        return
    with st.container(horizontal=True, gap="small"):
        for track_id, song, artist in playable:
            with st.container(width=320):
                st.caption(f"{song} — {artist}")
                _show_player(track_id, song, artist)


def _clean(value: object) -> str:
    return str(value).strip().lower()


def _artist_tokens(value: object) -> set[str]:
    return {token for token in _TOKEN.findall(_clean(value)) if token not in _ARTIST_STOPWORDS}


def _match_score(distance: np.ndarray) -> np.ndarray:
    """Turn feature distance into a 0–100 similarity score."""
    return np.rint(100 / (1 + distance)).astype(int)


@st.cache_data
def load_billboard(path_str: str) -> pd.DataFrame:
    raw = pd.read_csv(path_str)
    lookup = {column.lower(): column for column in raw.columns}

    def column(*names: str) -> pd.Series:
        for name in names:
            if name in lookup:
                return raw[lookup[name]]
        raise KeyError(names[0])

    chart = pd.DataFrame(
        {
            "rank": pd.to_numeric(column("rank"), errors="coerce"),
            "song": column("title", "song").astype(str).str.strip(),
            "artist": column("artist").astype(str).str.strip(),
        }
    )
    chart = chart.dropna(subset=["rank", "song", "artist"])
    chart = chart[chart["song"].ne("") & chart["artist"].ne("")]
    chart["rank"] = chart["rank"].astype(int)
    return chart.sort_values("rank").drop_duplicates("rank").reset_index(drop=True)


@st.cache_data
def load_songs(path_str: str) -> pd.DataFrame:
    songs = pd.read_csv(path_str)
    missing = [feature for feature in AUDIO_FEATURES + ["song", "artist"] if feature not in songs.columns]
    if missing:
        raise KeyError(", ".join(missing))

    songs = songs.dropna(subset=["song", "artist", *AUDIO_FEATURES]).copy()
    songs["song"] = songs["song"].astype(str).str.strip()
    songs["artist"] = songs["artist"].astype(str).str.strip()
    if "genre" not in songs.columns:
        songs["genre"] = "Unknown"
    songs["genre"] = songs["genre"].fillna("Unknown").astype(str)
    songs = songs[songs["song"].ne("") & songs["artist"].ne("")].reset_index(drop=True)
    songs["_song_key"] = songs["song"].map(_clean)
    songs["_artist_key"] = songs["artist"].map(_clean)
    return songs


def _same_genre_rate(
    genres: np.ndarray,
    clusters: np.ndarray,
    distances: np.ndarray,
    track_keys: np.ndarray,
    method: str,
    n: int = 5,
) -> float:
    """Average share of top-n picks that match the input song's genre."""
    scores: list[float] = []
    for idx in range(len(genres)):
        if method == METHOD_CLUSTER:
            pool = np.flatnonzero(clusters == clusters[idx])
        else:
            pool = np.arange(len(genres))
        pool = pool[pool != idx]
        if len(pool) == 0:
            continue

        ordered = pool[np.argsort(distances[idx, pool], kind="mergesort")]
        picked: list[int] = []
        seen = {tuple(track_keys[idx])}
        for candidate in ordered:
            key = tuple(track_keys[int(candidate)])
            if key in seen:
                continue
            seen.add(key)
            picked.append(int(candidate))
            if len(picked) == n:
                break
        if picked:
            scores.append(float(np.mean(genres[picked] == genres[idx])))
    return float(np.mean(scores)) if scores else 0.0


@st.cache_resource
def build_sound_model(path_str: str, n_clusters: int) -> SoundModel:
    songs = load_songs(path_str).copy()
    features = [feature for feature in AUDIO_FEATURES if feature in songs.columns]
    scaled = StandardScaler().fit_transform(songs[features])
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)
    songs["cluster"] = labels.astype(int)
    distances = pairwise_distances(scaled)
    track_keys = np.column_stack([songs["_song_key"].to_numpy(), songs["_artist_key"].to_numpy()])
    genres = songs["genre"].to_numpy()
    hit_rates = {
        method: _same_genre_rate(genres, songs["cluster"].to_numpy(), distances, track_keys, method)
        for method in (METHOD_CLUSTER, METHOD_ALL)
    }
    sizes = songs["cluster"].value_counts().sort_index().to_dict()
    return SoundModel(
        songs=songs,
        distances=distances,
        track_keys=track_keys,
        features=features,
        silhouette=float(silhouette_score(scaled, labels)),
        hit_rates=hit_rates,
        cluster_sizes={int(cluster): int(size) for cluster, size in sizes.items()},
    )


def chart_recommendations(chart: pd.DataFrame, idx: int, n: int, mode: str) -> pd.DataFrame:
    selected_rank = int(chart.at[idx, "rank"])
    others = chart.drop(index=idx)
    if mode == MODE_TOP:
        picked = others.sort_values("rank")
    else:
        gap = (others["rank"] - selected_rank).abs()
        picked = others.assign(_gap=gap).sort_values(["_gap", "rank"])

    picked = picked.head(n).copy()
    picked["vs_pick"] = selected_rank - picked["rank"]
    return picked[["rank", "song", "artist", "vs_pick"]].reset_index(drop=True)


def best_catalog_match(songs: pd.DataFrame, title: str, artist: str) -> tuple[int, float] | None:
    hits = songs.index[songs["_song_key"].eq(_clean(title))].tolist()
    if not hits:
        return None

    chart_tokens = _artist_tokens(artist)

    def overlap(row_id: int) -> float:
        catalog_tokens = _artist_tokens(songs.at[row_id, "artist"])
        if not chart_tokens or not catalog_tokens:
            return 0.0
        return len(chart_tokens & catalog_tokens) / len(chart_tokens | catalog_tokens)

    ranked = sorted(hits, key=overlap, reverse=True)
    return ranked[0], overlap(ranked[0])


def sound_recommendations(
    model: SoundModel,
    idx: int,
    n: int,
    method: str,
    skip_same_artist: bool,
    same_genre_only: bool,
) -> tuple[pd.DataFrame, int]:
    songs = model.songs
    cluster = songs["cluster"].to_numpy()
    target_cluster = int(cluster[idx])
    mask = np.ones(len(songs), dtype=bool)
    mask &= ~np.all(model.track_keys == model.track_keys[idx], axis=1)
    if skip_same_artist:
        mask &= songs["_artist_key"].to_numpy() != songs.at[idx, "_artist_key"]
    if same_genre_only:
        mask &= songs["genre"].to_numpy() == songs.at[idx, "genre"]

    pool = np.flatnonzero(mask)
    if method == METHOD_CLUSTER:
        primary = pool[cluster[pool] == target_cluster]
        backup = pool[cluster[pool] != target_cluster]
    else:
        primary = pool
        backup = np.empty(0, dtype=int)

    def by_distance(row_ids: np.ndarray) -> np.ndarray:
        if len(row_ids) == 0:
            return row_ids
        return row_ids[np.argsort(model.distances[idx, row_ids], kind="mergesort")]

    picked: list[int] = []
    seen: set[tuple[str, str]] = set()
    outside = 0
    for candidate in list(by_distance(primary)) + list(by_distance(backup)):
        row_id = int(candidate)
        key = tuple(model.track_keys[row_id])
        if key in seen:
            continue
        seen.add(key)
        picked.append(row_id)
        if method == METHOD_CLUSTER and int(cluster[row_id]) != target_cluster:
            outside += 1
        if len(picked) == n:
            break

    columns = ["song", "artist", "genre", "match", "distance", "same_genre", "same_cluster", "href", "track_id"]
    if not picked:
        return pd.DataFrame(columns=columns), 0

    picked_ids = np.array(picked)
    distances = model.distances[idx, picked_ids]
    recs = songs.loc[picked_ids, ["song", "artist", "genre"]].copy()
    recs["match"] = _match_score(distances)
    recs["distance"] = distances
    recs["same_genre"] = np.where(songs.loc[picked_ids, "genre"].to_numpy() == songs.at[idx, "genre"], "Yes", "No")
    recs["same_cluster"] = np.where(cluster[picked_ids] == target_cluster, "Yes", "No")
    recs["href"] = songs.loc[picked_ids, "href"] if "href" in songs.columns else pd.NA
    if "id" in songs.columns:
        recs["track_id"] = songs.loc[picked_ids, "id"].map(_spotify_track_id).to_numpy()
    else:
        recs["track_id"] = recs["href"].map(_spotify_track_id)
    recs["_row"] = picked_ids
    return recs.reset_index(drop=True), outside


def _open_catalog(row_id: int) -> None:
    st.session_state["view"] = VIEW_SOUND
    st.session_state["catalog_pick"] = row_id


def _chart_label(chart: pd.DataFrame, idx: int) -> str:
    row = chart.loc[idx]
    return f"#{int(row['rank'])}  {row['song']} — {row['artist']}"


def _catalog_label(songs: pd.DataFrame, idx: int) -> str:
    row = songs.loc[idx]
    return f"{row['song']} — {row['artist']}"


def _render_chart(chart: pd.DataFrame, catalog: pd.DataFrame | None, catalog_error: str | None) -> None:
    st.subheader("Hot 100", icon=":material/leaderboard:")
    st.caption(
        "What's popular on the chart right now, no taste profile needed. "
        "Pick nearby ranks to stay around your choice, or top of the chart to jump straight to #1."
    )

    pick_col, settings_col = st.columns([1.3, 1], vertical_alignment="bottom")
    with pick_col:
        idx = st.selectbox(
            "Chart song",
            options=chart.index.tolist(),
            format_func=lambda row_id: _chart_label(chart, row_id),
            key="chart_pick",
        )
    with settings_col:
        mode = st.segmented_control(
            "Recommendation rule",
            options=[MODE_NEIGHBORS, MODE_TOP],
            default=MODE_NEIGHBORS,
            format_func=lambda value: "Nearby ranks" if value == MODE_NEIGHBORS else "Top of the chart",
            required=True,
            key="chart_mode",
        )
        n_songs = st.slider("How many", min_value=1, max_value=10, value=5, key="chart_n")

    selected = chart.loc[idx]
    match = None if catalog is None else best_catalog_match(catalog, selected["song"], selected["artist"])

    position, from_top, in_catalog = st.columns(3)
    position.metric("Position", f"#{int(selected['rank'])}")
    from_top.metric("Places from #1", int(selected["rank"]) - 1)
    in_catalog.metric("In Gnod catalog", "Yes" if match else "No")

    if catalog is None:
        st.caption(catalog_error or "Gnod catalog isn't loaded.")
    elif match is None:
        st.caption("Not in the Gnod catalog, so you're just seeing chart picks here.")
    else:
        row_id, artist_score = match
        catalog_row = catalog.loc[row_id]
        with st.container(border=True):
            if artist_score > 0:
                st.markdown(f"Found **{catalog_row['song']}** by {catalog_row['artist']} in Gnod.")
            else:
                st.markdown(
                    f"Found **{catalog_row['song']}** by {catalog_row['artist']} in Gnod "
                    "(same title, different artist)."
                )
            st.button(
                "Find songs that sound like it",
                icon=":material/graphic_eq:",
                on_click=_open_catalog,
                args=(int(row_id),),
                key="open_catalog",
            )

    recs = chart_recommendations(chart, int(idx), int(n_songs), mode or MODE_NEIGHBORS)
    st.dataframe(
        recs,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d"),
            "song": st.column_config.TextColumn("Song"),
            "artist": st.column_config.TextColumn("Artist"),
            "vs_pick": st.column_config.NumberColumn(
                "Vs. your pick",
                format="%+d",
                help="Positive means higher on the chart than your pick.",
            ),
        },
        hide_index=True,
    )

    with st.expander("Full Hot 100", icon=":material/table_chart:"):
        st.dataframe(
            chart,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "song": st.column_config.TextColumn("Song"),
                "artist": st.column_config.TextColumn("Artist"),
            },
            hide_index=True,
        )


def _render_sound(catalog: pd.DataFrame) -> None:
    st.subheader("Sound match", icon=":material/graphic_eq:")
    with st.spinner("Scoring the catalog…"):
        model = build_sound_model(str(CATALOG_PATH), N_CLUSTERS)

    songs = model.songs
    st.caption(
        f"Uses 9 scaled audio features and {N_CLUSTERS} K-Means clusters (no key or mode, so songs aren't split "
        "just by major vs. minor). Higher match means closer in sound."
    )

    select_kwargs: dict[str, int | None] = {}
    if "catalog_pick" not in st.session_state:
        select_kwargs["index"] = None
    idx = st.selectbox(
        "Filter by song or artist",
        options=[int(row_id) for row_id in songs.index],
        format_func=lambda row_id: _catalog_label(songs, row_id),
        placeholder="Type to search...",
        icon=":material/search:",
        filter_mode="contains",
        key="catalog_pick",
        **select_kwargs,
    )
    if idx is None:
        st.caption("Type a song or artist to pick a track.")
        _render_model_notes(model)
        return

    method_col, count_col = st.columns([1.2, 1], vertical_alignment="bottom")
    with method_col:
        method = st.segmented_control(
            "Where to look",
            options=[METHOD_CLUSTER, METHOD_ALL],
            default=METHOD_CLUSTER,
            format_func=lambda value: "Same cluster" if value == METHOD_CLUSTER else "Whole catalog",
            required=True,
            width="stretch",
            key="sound_method",
        )
    with count_col:
        n_songs = st.slider("How many", min_value=1, max_value=12, value=5, key="sound_n")
    with st.container(horizontal=True, vertical_alignment="center"):
        skip_same_artist = st.toggle("Skip this artist", value=False, key="skip_artist")
        same_genre_only = st.toggle("Same genre only", value=False, key="same_genre")

    picked = songs.loc[idx]
    cluster_id = int(picked["cluster"])
    recs, outside = sound_recommendations(
        model,
        int(idx),
        int(n_songs),
        method or METHOD_CLUSTER,
        bool(skip_same_artist),
        bool(same_genre_only),
    )
    genre_hits = int((recs["same_genre"] == "Yes").sum()) if not recs.empty else 0
    hidden = [column for column in ("_row", "track_id") if column in recs.columns]
    shown = recs.drop(columns=hidden) if not recs.empty else recs

    with st.container(border=True):
        title_col, link_col = st.columns([4, 1], vertical_alignment="center")
        with title_col:
            st.subheader(str(picked["song"]))
            st.caption(f"{picked['artist']} · {picked['genre']}")
        href = picked["href"] if "href" in songs.columns else None
        if isinstance(href, str) and href.startswith("http"):
            with link_col:
                st.link_button("Spotify", href, icon=":material/open_in_new:")
        selected_track = _spotify_track_id(picked["id"]) if "id" in songs.columns else None
        if selected_track is None:
            selected_track = _spotify_track_id(href)
        if selected_track:
            _show_player(selected_track, str(picked["song"]), str(picked["artist"]))

        cluster_col, size_col, tempo_col, hits_col = st.columns(4)
        cluster_col.metric("Cluster", cluster_id)
        size_col.metric("Songs in cluster", model.cluster_sizes.get(cluster_id, 0))
        tempo_col.metric("Tempo", f"{float(picked['tempo']):.0f} bpm")
        hits_col.metric("Same genre", f"{genre_hits} of {len(shown)}" if not shown.empty else "—")

    if recs.empty:
        st.warning("Nothing left with those filters—try turning one off.")
        return

    if outside:
        st.caption(f"{outside} pick(s) came from outside cluster {cluster_id} because there weren't enough left inside it.")

    st.dataframe(
        shown,
        column_config={
            "song": st.column_config.TextColumn("Song"),
            "artist": st.column_config.TextColumn("Artist"),
            "genre": st.column_config.TextColumn("Genre"),
            "match": st.column_config.ProgressColumn("Match", min_value=0, max_value=100, format="%d"),
            "distance": st.column_config.NumberColumn("Distance", format="%.2f"),
            "same_genre": st.column_config.TextColumn("Same genre"),
            "same_cluster": st.column_config.TextColumn("Same cluster"),
            "href": st.column_config.LinkColumn("Listen", display_text="Open"),
        },
        hide_index=True,
    )
    _render_players(
        [
            (str(row.track_id), str(row.song), str(row.artist))
            for row in recs.itertuples(index=False)
            if _spotify_track_id(row.track_id)
        ]
    )

    profile_features = [feature for feature in UNIT_FEATURES if feature in songs.columns]
    if profile_features and not recs.empty:
        compare_id = int(recs.iloc[0]["_row"])
        profile = pd.DataFrame(
            [
                {"feature": feature, "Song": "Your pick", "level": float(songs.at[int(idx), feature])}
                for feature in profile_features
            ]
            + [
                {"feature": feature, "Song": "Closest match", "level": float(songs.at[compare_id, feature])}
                for feature in profile_features
            ]
        )
        st.bar_chart(
            profile,
            x="feature",
            y="level",
            x_label="Feature",
            y_label="0 to 1",
            color="Song",
            horizontal=True,
            stack=False,
            sort=False,
            height=360,
        )
        st.caption("Showing 0–1 features only (tempo and loudness are on different scales).")

    _render_model_notes(model)


def _render_model_notes(model: SoundModel) -> None:
    with st.expander("How the clusters score", icon=":material/analytics:"):
        st.caption(
            f"Silhouette is {model.silhouette:.2f}—fairly low because genres and vibes overlap "
            "(especially hybrid styles like reggae, hip-hop, funk, and folk)."
        )
        rates = pd.DataFrame(
            {
                "method": ["Same cluster", "Whole catalog"],
                "same_genre_rate": [model.hit_rates[METHOD_CLUSTER], model.hit_rates[METHOD_ALL]],
            }
        )
        st.dataframe(
            rates,
            column_config={
                "method": st.column_config.TextColumn("Method"),
                "same_genre_rate": st.column_config.NumberColumn("Same-genre rate", format="percent"),
            },
            hide_index=True,
        )
        st.caption(
            "Closest picks hit around 26–27% same-genre (vs. ~15% random in cluster and ~6% random overall). "
            "Clusters still help group vibes and browse beyond just the nearest neighbors."
        )


st.title("Song recommender", icon=":material/music_note:")
st.caption("Popular songs on the Billboard chart right now, or Gnod tracks that sound like what you picked.")

try:
    chart = load_billboard(str(CHART_PATH))
except FileNotFoundError:
    st.error("Add `billboard_hot100.csv` next to `app.py`.", title="Chart file missing", icon=":material/error:")
    st.stop()
except KeyError as exc:
    st.error(f"The chart file needs a {exc} column.", title="Chart file unusable", icon=":material/error:")
    st.stop()
except pd.errors.ParserError:
    st.error("`billboard_hot100.csv` is not a readable CSV.", title="Chart file unreadable", icon=":material/error:")
    st.stop()

catalog_error: str | None = None
try:
    catalog = load_songs(str(CATALOG_PATH))
except FileNotFoundError:
    catalog = None
    catalog_error = "Add `gnod_songs.csv` next to `app.py` to use sound matching."
except KeyError as exc:
    catalog = None
    catalog_error = f"The catalog is missing {exc}."
except pd.errors.ParserError:
    catalog = None
    catalog_error = "`gnod_songs.csv` is not a readable CSV."

with st.sidebar:
    st.header("How it works", icon=":material/info:")
    st.markdown("**Chart:** Picks popular tracks near your rank or from the top of the Hot 100.")
    st.markdown("**Sound match:** Compares 9 scaled audio features across 6 K-Means clusters, skipping repeat song + artist pairs.")

if "view" not in st.session_state:
    st.session_state["view"] = VIEW_CHART

view = st.segmented_control(
    "Source",
    options=[VIEW_CHART, VIEW_SOUND],
    format_func=lambda value: (
        ":material/leaderboard: Chart" if value == VIEW_CHART else ":material/graphic_eq: Sound match"
    ),
    required=True,
    key="view",
    width="stretch",
)

if view == VIEW_SOUND:
    if catalog is None:
        st.error(
            catalog_error or "The sound catalog is unavailable.",
            title="Catalog unavailable",
            icon=":material/error:",
        )
    else:
        _render_sound(catalog)
else:
    _render_chart(chart, catalog, catalog_error)