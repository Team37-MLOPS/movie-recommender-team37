"""Loading and splitting utilities for the MovieLens 1M ratings dataset."""
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RATINGS_PATH = PROJECT_ROOT / "ml-1m" / "ratings.dat"

RELEVANCE_THRESHOLD = 4  # rating >= this counts as "liked" for Precision/Recall


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
