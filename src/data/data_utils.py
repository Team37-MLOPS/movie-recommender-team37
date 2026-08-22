"""Loading and splitting utilities for the MovieLens 1M ratings dataset."""
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RATINGS_PATH = PROJECT_ROOT / "data" / "raw" / "ml-1m" / "ratings.dat"

RELEVANCE_THRESHOLD = 4  # rating >= this counts as "liked" for Precision/Recall

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def _genre_column_name(genre: str) -> str:
    return f"genre_{genre.lower().replace('-', '_').replace(chr(39), '')}"


CONTENT_FEATURE_COLUMNS = (
    ["gender_is_male", "age", "occupation"]
    + ["user_rating_count", "user_mean_rating", "user_rating_stddev"]
    + ["movie_rating_count", "movie_mean_rating", "movie_rating_stddev"]
    + ["num_genres"]
    + [_genre_column_name(g) for g in GENRES]
)


def load_ratings(path: Path = RATINGS_PATH) -> pd.DataFrame:
    """Read the raw `UserID::MovieID::Rating::Timestamp` ratings file."""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

#df = load_ratings()
#df.head


def train_val_test_split_by_user(
    df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal split, sorted by timestamp: train / val / test."""
    df = df.sort_values(["user_id", "timestamp"])
    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby("user_id", sort=False):
        n = len(group)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_train = max(1, n - n_val - n_test)
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train : n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val :])
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, val_df, test_df


def build_user_items_map(train_df: pd.DataFrame) -> Dict[int, set]:
    """user_id -> set of movie_ids already rated in training data (excluded
    from that user's recommendations)."""
    return train_df.groupby("user_id")["movie_id"].apply(set).to_dict()


def build_relevant_items_map(test_df: pd.DataFrame, threshold: int = RELEVANCE_THRESHOLD) -> Dict[int, set]:
    """user_id -> set of movie_ids the user rated >= threshold in the test
    set (i.e. the ground-truth "liked" items for Precision@K/Recall@K)."""
    relevant = test_df[test_df["rating"] >= threshold]
    return relevant.groupby("user_id")["movie_id"].apply(set).to_dict()


def load_engineered_features(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the Spark-preprocessed `movies` and `users` parquet outputs and
    reshape them into static, per-entity content features (genre one-hot,
    demographics) for use by content-aware models like the XGBoost hybrid.

    Deliberately excludes the `user_features`/`movie_features` parquet
    outputs Spark also writes (per-user/movie rating aggregates): those are
    computed over *all* cleaned ratings, so joining them directly onto a
    train/test split would leak test-set signal into training rows. Callers
    that need rating aggregates should compute their own train-only
    per-user/per-movie stats instead (see `XGBHybridModel.fit`).
    """
    movies = pd.read_parquet(processed_dir / "movies")[["movie_id", "genre_array"]].copy()
    for genre in GENRES:
        column = _genre_column_name(genre)
        movies[column] = movies["genre_array"].apply(lambda genres, g=genre: int(g in genres))
    movies["num_genres"] = movies["genre_array"].apply(len)
    movies = movies.drop(columns=["genre_array"])

    users = pd.read_parquet(processed_dir / "users")[["user_id", "gender", "age", "occupation"]].copy()
    users["gender_is_male"] = (users["gender"] == "M").astype(int)
    users = users.drop(columns=["gender"])

    return movies, users
