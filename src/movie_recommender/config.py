from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class Paths(BaseModel):
    dataset_zip: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    model_dir: Path
    artifact_dir: Path
    recommendation_path: Path
    popularity_path: Path
    metrics_path: Path


class SparkSettings(BaseModel):
    app_name: str
    master: str
    shuffle_partitions: int


class TrainingSettings(BaseModel):
    sample_fraction: float
    implicit_prefs: bool
    rank: int
    max_iter: int
    reg_param: float
    top_k: int
    train_ratio: float
    cold_start_strategy: str


class MLflowSettings(BaseModel):
    tracking_uri: str
    experiment_name: str


class ApiSettings(BaseModel):
    default_k: int
    max_k: int


class Settings(BaseModel):
    project_name: str
    random_seed: int
    paths: Paths
    spark: SparkSettings
    training: TrainingSettings
    mlflow: MLflowSettings
    api: ApiSettings


@lru_cache
def load_settings(config_path: str | Path = "configs/settings.yaml") -> Settings:
    with Path(config_path).open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return Settings.model_validate(data)
