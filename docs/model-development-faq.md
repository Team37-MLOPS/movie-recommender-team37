# Model Development FAQ

This FAQ explains how the pipeline behaves with new data, how to add a new
recommendation model, and how team members can compare metrics before opening a
pull request.

## What Happens When New Data Is Loaded?

The project currently uses a batch pipeline. When the dataset changes, rerun the
pipeline from extraction through training.

Local command flow:

```bash
make extract
make preprocess
make train
```

Fast sample flow:

```bash
python scripts/run_sample_pipeline.py
```

Airflow flow:

```text
Trigger DAG: movie_recommender_pipeline
```

The pipeline creates or refreshes:

```text
data/raw/ml-1m/
data/processed/
models/recommendations.parquet
models/popular_movies.parquet
models/als_model/
artifacts/metrics.json
mlruns/
```

After retraining, restart the API so it reloads model artifacts:

```bash
docker compose restart api
```

or, if running without Docker, stop and restart:

```bash
make api
```

## Where Are Metrics Stored?

The training pipeline writes summary metrics to:

```text
artifacts/metrics.json
```

MLflow also records parameters, metrics, and artifacts. If running locally:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

If using Docker Compose, MLflow is available at:

```text
http://localhost:5000
```

## What Metrics Should Be Checked?

At minimum, compare:

- `als_rmse`: lower is better
- `als_mae`: lower is better
- `als_precision_at_10`: higher is better

For any new model, keep metrics comparable. Recommended common metrics:

- RMSE
- MAE
- precision@k
- recommendation coverage
- training time
- number of users served without fallback

## How Do I Add A New Model?

Use a feature branch:

```bash
git checkout main
git pull
git checkout -b <username>/model-<model-name>
```

Add the model implementation under:

```text
src/movie_recommender/models/
```

Example:

```text
src/movie_recommender/models/item_item.py
```

The model should:

1. Read processed data from `data/processed/`.
2. Train using the same train/test split seed where possible.
3. Generate recommendations in a consistent schema.
4. Evaluate using comparable metrics.
5. Log metrics to MLflow.
6. Save artifacts under `models/`.
7. Update `artifacts/metrics.json`.

## What Recommendation Output Schema Should A New Model Produce?

The API expects recommendation artifacts to contain fields compatible with:

```text
user_id
movie_id
rank
score
clean_title or title
genres
release_year
```

The serving layer reads:

```text
models/recommendations.parquet
models/popular_movies.parquet
```

If a new model writes a different file, either:

- adapt the training step to export to the existing paths, or
- update the serving repository carefully and add tests.

For easiest integration, keep writing:

```text
models/recommendations.parquet
```

## How Do I Wire A New Model Into Training?

The current main training entrypoint is:

```text
src/movie_recommender/models/train.py
```

For a first version, add a function such as:

```python
def train_item_item(...):
    ...
```

Then call it from `train()` and add its metrics to the metrics dictionary:

```python
metrics = {
    "als_rmse": als_rmse,
    "als_mae": als_mae,
    "als_precision_at_10": als_precision_at_k,
    "item_item_precision_at_10": item_item_precision_at_k,
}
```

Also log parameters and metrics to MLflow:

```python
mlflow.log_params(...)
mlflow.log_metrics(metrics)
```

## How Do I Run My New Model Locally?

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Run the full local pipeline:

```bash
make extract
make preprocess
make train
```

Or run the deterministic sample pipeline:

```bash
python scripts/run_sample_pipeline.py
```

Check metrics:

```bash
cat artifacts/metrics.json
```

Start MLflow:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

## How Do I Compare My Model With Existing Models?

Use the same dataset, split, and `random_seed`.

Check:

```bash
cat artifacts/metrics.json
```

Example comparison:

```json
{
  "als_rmse": 0.86,
  "als_mae": 0.68,
  "als_precision_at_10": 0.04,
  "item_item_precision_at_10": 0.05
}
```

Interpretation:

- If RMSE/MAE improves, rating prediction quality improved.
- If precision@k improves, top-k recommendation relevance improved.
- If training time is much higher, decide whether the improvement is worth the
  cost.
- If fallback ratio increases during serving, the model may not cover enough
  users.

## How Do I Test The API With My Model?

After training, start the API:

```bash
make api
```

Check readiness:

```bash
curl http://localhost:8000/ready
```

Request recommendations:

```bash
curl "http://localhost:8000/recommendations/1?k=10"
```

Open Swagger:

```text
http://localhost:8000/docs
```

If running Docker Compose:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000/docs
```

## How Do I Check Serving Metrics?

Generate traffic:

```bash
for i in $(seq 1 30); do
  curl "http://localhost:8000/recommendations/1?k=5" >/dev/null
  curl "http://localhost:8000/recommendations/99999?k=5" >/dev/null
  sleep 1
done
```

Open Grafana:

```text
http://localhost:3000/d/efvbt8hwrwruoe/basic-api-service-dashboard
```

Check:

- prediction request rate
- fallback ratio
- recommendation score distribution
- recommended movie release-year mix
- API latency

## What Should I Run Before Opening A PR?

Run:

```bash
ruff check src api dags tests scripts
pytest -q
python -m py_compile dags/movie_recommender_pipeline.py
python scripts/run_sample_pipeline.py
```

If Docker files or serving behavior changed:

```bash
docker compose up --build
curl http://localhost:8000/ready
curl "http://localhost:8000/recommendations/1?k=10"
```

## What Should The PR Include?

Include:

- model name and approach
- files changed
- metrics before and after
- MLflow run ID or screenshot
- whether API output schema changed
- whether serving metrics changed
- known limitations

Example PR summary:

```text
## Summary
- Add item-item collaborative filtering model
- Compare item-item against ALS and popularity baseline
- Log item-item precision@10 to MLflow and metrics.json

## Metrics
- ALS precision@10: 0.04
- Item-item precision@10: 0.05
- ALS RMSE: 0.86
- Item-item RMSE: not applicable

## Validation
- ruff check passed
- pytest passed
- sample pipeline passed
- API recommendation endpoint tested locally
```

## What If My Model Needs Different Features?

Add feature engineering in:

```text
src/movie_recommender/data/preprocess_spark.py
```

Keep existing output columns unless there is a strong reason to change them.
If new columns are added, document them and add tests.

If the preprocessing output changes, rerun:

```bash
make preprocess
make train
```

## What If My Model Is Experimental And Should Not Replace The Current API Model?

Write artifacts to a model-specific directory:

```text
models/experiments/<model-name>/
```

Still log metrics to MLflow. Do not change:

```text
models/recommendations.parquet
models/popular_movies.parquet
```

until the model is selected for serving.

## Recommended Future Improvement

Introduce a model registry pattern:

```text
src/movie_recommender/models/
  base.py
  registry.py
  popularity.py
  als.py
  item_item.py
```

Then configure models in `params.yaml`:

```yaml
models:
  enabled:
    - popularity
    - als
    - item_item
```

This would let each team member add a model with minimal changes to the shared
training pipeline.

