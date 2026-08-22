from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    title: str
    genres: str
    score: float
    rank: int
    release_year: int | None


class RecommendationRepository:
    def __init__(self, recommendation_path: Path, popularity_path: Path) -> None:
        self.recommendation_path = recommendation_path
        self.popularity_path = popularity_path
        self._recommendations = self._read_parquet_dir(recommendation_path)
        self._popular = self._read_parquet_dir(popularity_path)

    @staticmethod
    def _read_parquet_dir(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    @property
    def ready(self) -> bool:
        return not self._popular.empty

    def movie_metadata(self, movie_id: int) -> dict:
        """Best-effort title/genres/release_year lookup by movie_id, used
        to enrich recommendation sources (e.g. a registry model) that
        return bare movie_id lists with no metadata of their own."""
        for frame in (self._popular, self._recommendations):
            if frame.empty or "movie_id" not in frame.columns:
                continue
            rows = frame[frame["movie_id"] == movie_id]
            if not rows.empty:
                row = rows.iloc[0]
                release_year = row.get("release_year")
                if pd.isna(release_year):
                    release_year = None
                return {
                    "title": str(row.get("clean_title") or row.get("title") or ""),
                    "genres": str(row.get("genres") or ""),
                    "release_year": int(release_year) if release_year is not None else None,
                }
        return {"title": "", "genres": "", "release_year": None}

    def recommend(self, user_id: int, k: int) -> tuple[list[Recommendation], bool]:
        if not self._recommendations.empty and "user_id" in self._recommendations.columns:
            user_rows = self._recommendations[self._recommendations["user_id"] == user_id]
            if not user_rows.empty:
                return self._to_recommendations(user_rows.sort_values("rank").head(k)), False

        fallback = self._popular.head(k).copy()
        fallback["rank"] = range(1, len(fallback) + 1)
        return self._to_recommendations(fallback), True

    @staticmethod
    def _to_recommendations(frame: pd.DataFrame) -> list[Recommendation]:
        items: list[Recommendation] = []
        for _, row in frame.iterrows():
            title = row.get("clean_title") or row.get("title") or ""
            release_year = row.get("release_year")
            if pd.isna(release_year):
                release_year = None
            items.append(
                Recommendation(
                    movie_id=int(row["movie_id"]),
                    title=str(title),
                    genres=str(row.get("genres") or ""),
                    score=float(row.get("score") or row.get("mean_rating") or 0.0),
                    rank=int(row.get("rank") or len(items) + 1),
                    release_year=int(release_year) if release_year is not None else None,
                )
            )
        return items
