# Technical Report

## Problem Statement

The goal is to build an end-to-end MLOps pipeline for a movie recommendation
system. The system should process MovieLens data, train and compare
recommendation models, version artifacts, deploy a serving API, monitor runtime
behavior, and validate changes through CI/CD.

The recommendation API must support personalized recommendations for known users
and provide a fallback strategy for users without personalized model output.

## Dataset Description

The project uses the MovieLens 1M dataset from `ml-1m.zip`. The dataset contains
movie ratings, user information, and movie metadata. The main inputs are:

- ratings: user, movie, rating, and timestamp interactions
- movies: movie title and genre metadata
- users: demographic information available in the original dataset

The current pipeline focuses on ratings and movie metadata. Preprocessing
extracts useful fields such as clean movie titles, genres, and release years.
The data is converted into Parquet outputs for repeatable downstream training
and serving.

## System Architecture

The architecture combines offline training and online serving:

- Airflow orchestrates the pipeline.
- Spark performs data cleaning, feature generation, and preprocessing.
- Model training produces recommendation artifacts.
- MLflow tracks parameters, metrics, and generated artifacts.
- DVC is used for dataset and artifact versioning.
- FastAPI serves recommendation requests.
- Prometheus collects API and prediction metrics.
- Grafana visualizes API health, prediction behavior, and lightweight drift
  signals.
- GitHub Actions runs CI checks and pipeline validation.

The high-level flow is:

```text
ml-1m.zip
  -> Airflow DAG
  -> extraction
  -> Spark preprocessing
  -> processed Parquet data
  -> model training
  -> MLflow metrics and model artifacts
  -> FastAPI serving
  -> Prometheus metrics
  -> Grafana dashboard
```

## Data Pipeline

The data pipeline has three main stages:

1. Extraction: unzip `ml-1m.zip` into `data/raw/ml-1m/`.
2. Preprocessing: use Spark to clean and transform raw MovieLens files into
   processed Parquet datasets under `data/processed/`.
3. Training: load processed Parquet files, split train/test data, train models,
   evaluate metrics, and write artifacts under `models/` and `artifacts/`.

Airflow defines the orchestration DAG:

```text
dags/movie_recommender_pipeline.py
```

For quick validation, the repository also includes:

```text
scripts/run_sample_pipeline.py
```

## Model Development

The project currently supports:

- popularity baseline recommendations
- ALS collaborative filtering recommendations

The popularity baseline provides a simple fallback and benchmark. ALS provides
personalized collaborative filtering recommendations for users present in the
training data.

Model metrics are written to:

```text
artifacts/metrics.json
```

Experiments are tracked in MLflow. New model development guidance is documented
in:

```text
docs/model-development-faq.md
```

## Deployment Strategy

The primary deployment path for the project is Docker Compose. It runs:

- FastAPI
- Airflow webserver and scheduler
- MLflow
- Postgres
- Prometheus
- Grafana

This keeps the demo reproducible on local systems without requiring paid cloud
infrastructure.

Kubernetes manifests are included for local `kind` deployment of API and
monitoring components. AWS deployment is documented as a planned/free-tier-aware
path rather than always-on infrastructure, because always-running cloud
resources can incur cost.

## Monitoring Strategy

The FastAPI service exposes Prometheus metrics at `/metrics`. The monitoring
strategy covers:

- API request rate
- API latency
- prediction request rate
- fallback ratio
- recommendation item throughput
- recommendation score distribution
- release-year mix as a lightweight drift signal

Grafana dashboards are provisioned from source-controlled JSON so the dashboard
can be recreated on any local system when Docker Compose starts.

Detailed metric documentation is available in:

```text
docs/monitoring-metrics.md
```

## CI/CD Implementation

GitHub Actions validates changes through:

- dependency installation
- linting with Ruff
- unit and API tests with Pytest
- Airflow DAG syntax validation
- deterministic sample pipeline execution
- Docker image build
- metrics artifact upload

The full pipeline workflow can run the training pipeline and upload model
artifacts. The release workflow design supports branch-based releases for `test`
and `main`, where test releases can be treated as prereleases and main releases
as production releases.

## Results And Discussion

The pipeline produces trained recommendation artifacts and evaluation metrics.
The popularity model acts as a baseline and fallback, while ALS provides
personalized recommendations for known users.

The API can serve both personalized and fallback recommendations. Monitoring
helps explain whether traffic is mostly personalized or fallback-based. This is
important because a high fallback ratio may indicate that the model does not
cover enough current users or that test traffic is using unknown user IDs.

The current implementation is suitable for local demonstration and CI validation.
It demonstrates the complete MLOps lifecycle, while leaving production-grade
cloud deployment as a future extension.

## Challenges Faced

Key challenges included:

- Testing consistently across team members without shared always-on deployment
  infrastructure.
- Avoiding unnecessary cloud cost while still designing a realistic deployment
  path.
- Keeping generated data, model artifacts, MLflow runs, and Docker volumes out
  of Git while preserving reproducibility.
- Managing local port conflicts for services such as Airflow, Grafana,
  Prometheus, and FastAPI.
- Making dashboard provisioning reproducible across machines instead of relying
  only on manually edited Grafana dashboards.
- Ensuring CI can run the sample pipeline without requiring large external
  infrastructure.
- Balancing full MovieLens training time with quick sample runs needed for CI
  and PR validation.
- Keeping model output schemas stable so the FastAPI serving layer does not
  break when models are changed.

## Future Improvements

Planned improvements include:

- Kubernetes deployment for API and monitoring using the existing `k8s/`
  manifests as a starting point.
- AWS deployment pipeline for test and production stages, with automatic deploy
  on release and teardown workflows to avoid cost when not in use.
- S3-backed DVC remote for dataset and model artifact versioning.
- ECR-backed Docker image publishing for deployable API images.
- More formal model registry structure for adding new recommendation models.
- Stronger drift detection using stored baseline distributions and statistical
  distance metrics.
- Scheduled retraining when new data is available.
- Automated model promotion rules based on evaluation metrics.
- Better load testing for API latency and concurrency.

