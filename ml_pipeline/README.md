# MovieLens MLOps Pipeline (`ml_pipeline/`)

This folder implements the data pipeline, preprocessing, and baseline model
training for the movie-recommender project, using the MovieLens-1M dataset
(`ml-1m.zip` at the repo root).

## What's here

```
ml_pipeline/
├── data/
│   ├── raw/ml-1m/            # unzipped ratings.dat, movies.dat, users.dat
│   └── processed/            # Beam pipeline output (movielens_features.csv, gitignored)
├── data_pipeline/
│   └── beam_preprocess.py    # Apache Beam batch ETL: parse, clean, join, feature-engineer
├── airflow_dags/
│   └── movielens_pipeline_dag.py   # Airflow DAG: check data -> Beam -> validate -> train
├── training/
│   └── train_models.py       # Trains + compares 2 models, logs to MLflow
├── mlruns/                   # MLflow tracking store (gitignored; sqlite + artifacts)
└── requirements.txt
```

## Why these tools

- **Apache Airflow** (compulsory): orchestrates the pipeline as a 4-task DAG
  (`check_raw_data -> run_beam_preprocessing -> validate_processed_data -> train_baseline_models`),
  giving scheduling, retries, and a UI to trigger/inspect runs.
- **Apache Beam**: the raw dataset is three flat `.dat` files that need
  parsing, cleaning, and a two-way join (ratings + users + movies) before
  feature engineering. Beam expresses this as a portable parallel batch
  pipeline (running here on Beam's local Prism/Direct runner) that could be
  pointed at a distributed runner later without changing the transform
  logic — a good fit for this batch-ETL, non-streaming workload.
- **MLflow**: tracks params/metrics/artifacts for every training run so the
  two candidate models can be compared directly in the MLflow UI.

## Data processing (part 3)

`beam_preprocess.py` does, in order:

1. **Parsing**: reads `::`-delimited `ratings.dat` / `movies.dat` / `users.dat`
   (movie titles are Latin-1 encoded — handled with a custom coder).
2. **Cleaning**: drops malformed lines, drops ratings outside 1-5, de-dupes
   exact `(user_id, movie_id)` duplicate ratings.
3. **Missing values**: unknown/malformed gender codes are mapped to `"U"`
   instead of dropped; any row missing a join key is dropped.
4. **Feature engineering**: one-hot genre flags, age buckets, rating
   year/day-of-week from the timestamp, gender flag, `num_genres`.
5. **Join**: ratings ⋈ users ⋈ movies → one flat feature table
   (`data/processed/movielens_features.csv`, ~1.0M rows).

## Model development (part 4)

`training/train_models.py` trains and compares two structurally different
models for rating prediction (1-5 scale):

1. **SVD collaborative filtering** (`surprise.SVD`) — learns latent
   user/item factors from the ratings matrix alone.
2. **Gradient Boosting regression** (`sklearn.GradientBoostingRegressor`)
   on content + demographic features (genres, age/gender/occupation,
   per-user and per-movie historical rating stats computed on the train
   split only, to avoid leakage).

Both are evaluated with RMSE/MAE on a held-out 20% split and logged to
MLflow. Latest local run:

| Model | RMSE | MAE |
|---|---|---|
| SVD collaborative filtering | 0.872 | 0.685 |
| Gradient Boosting (content-based) | 0.916 | 0.725 |

## How to run

```bash
cd movie-recommender-team37
python3 -m venv .venv && source .venv/bin/activate
pip install -r ml_pipeline/requirements.txt

# unzip the dataset once (if not already present under ml_pipeline/data/raw/ml-1m)
unzip ml-1m.zip -d /tmp/ml1m && cp /tmp/ml1m/ml-1m/*.dat ml_pipeline/data/raw/ml-1m/

# 1. Run the Beam preprocessing job directly
python3 ml_pipeline/data_pipeline/beam_preprocess.py \
  --ratings ml_pipeline/data/raw/ml-1m/ratings.dat \
  --movies ml_pipeline/data/raw/ml-1m/movies.dat \
  --users ml_pipeline/data/raw/ml-1m/users.dat \
  --output ml_pipeline/data/processed/movielens_features

# 2. Train + compare models (logs to MLflow under ml_pipeline/mlruns)
python3 ml_pipeline/training/train_models.py \
  --data ml_pipeline/data/processed/movielens_features.csv

# 3. View experiment results in the MLflow UI
mlflow ui --backend-store-uri sqlite:///ml_pipeline/mlruns/mlflow.db

# 4. (Optional) Run the same 4 steps as one Airflow DAG
export AIRFLOW_HOME=~/airflow
airflow db init
cp ml_pipeline/airflow_dags/movielens_pipeline_dag.py $AIRFLOW_HOME/dags/
airflow dags trigger movielens_data_and_training_pipeline
```
