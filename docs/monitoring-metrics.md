# Monitoring Metrics

This document explains the Prometheus metrics exposed by the Movie Recommender
FastAPI service and how each Grafana dashboard panel should be interpreted.

The API exposes metrics at:

```text
http://localhost:8000/metrics
```

When running through Docker Compose, Prometheus scrapes the API internally at:

```text
http://api:8000/metrics
```

The local Grafana dashboard is provisioned from:

```text
docker/grafana/dashboards/api-dashboard.json
```

## Metric Types

### Counter

A counter is a value that only increases while the process is running. It resets
to zero when the API process restarts. Counters are usually queried with
`rate()` or `increase()`.

Example:

```promql
rate(movie_recommender_predictions_total[5m])
```

This converts the raw counter into events per second over the last 5 minutes.

### Histogram

A histogram records observations into buckets. Prometheus exposes histogram
metrics as several related series:

```text
<metric>_bucket
<metric>_count
<metric>_sum
```

The `_bucket` series has an `le` label. `le` means less than or equal to. For
example, `le="0.5"` means observations less than or equal to 0.5. The
`le="+Inf"` bucket contains all observations.

Histograms are used to calculate percentiles:

```promql
histogram_quantile(0.95, sum(rate(metric_bucket[5m])) by (le))
```

## API Metrics

### `movie_recommender_api_requests_total`

Type: counter

Labels:

- `endpoint`: logical API endpoint, such as `/health`, `/ready`, or
  `/recommendations/{user_id}`
- `status`: application-level status, such as `ok`, `ready`, `not_ready`, or
  `model_artifacts_missing`

What it measures:

This metric counts API requests handled by the application. It is incremented
inside each route after the route determines the application-level status.

Useful queries:

```promql
sum(rate(movie_recommender_api_requests_total[5m])) by (endpoint, status)
```

This shows request throughput per endpoint and status.

How to interpret:

- Higher `ok` rate means the API is receiving successful traffic.
- `not_ready` means the API was called before model artifacts were available.
- `model_artifacts_missing` on `/health` means the service is running, but the
  recommendation model files are missing.
- A sudden traffic drop may mean users stopped calling the service, the service
  is unavailable, or Prometheus is not scraping.

### `movie_recommender_api_request_latency_seconds`

Type: histogram

Labels:

- `endpoint`: logical API endpoint
- `le`: histogram bucket upper bound, automatically added by Prometheus

What it measures:

This metric records how long the recommendation endpoint takes to process a
request. The value is measured in seconds.

Useful queries:

```promql
histogram_quantile(
  0.95,
  sum(rate(movie_recommender_api_request_latency_seconds_bucket[5m])) by (le, endpoint)
)
```

This shows p95 latency. p95 means 95% of requests were faster than this value.

How to interpret:

- Low p95 latency means most requests are fast.
- High p95 latency means at least 5% of requests are slow.
- A rising p95 may indicate slower model artifact reads, container CPU pressure,
  filesystem pressure, or high request volume.
- If the panel has no data, generate recommendation traffic and wait for at
  least two Prometheus scrapes.

## Prediction Metrics

### `movie_recommender_predictions_total`

Type: counter

Labels:

- `fallback`: `true` if the API used popularity fallback recommendations;
  `false` if it used user-specific ALS recommendations

What it measures:

This metric counts recommendation API requests. It increments once per
successful `/recommendations/{user_id}` request.

Useful queries:

```promql
sum(rate(movie_recommender_predictions_total[5m])) by (fallback)
```

This shows prediction request rate split by fallback mode.

How to interpret:

- `fallback="false"` means the request returned personalized recommendations.
- `fallback="true"` means the user was unknown or had no personalized
  recommendations, so the API returned popularity-based recommendations.
- A high fallback rate may indicate user drift, missing user coverage, stale
  model artifacts, or traffic from many unseen users.

### `movie_recommender_recommendations_returned_total`

Type: counter

Labels:

- `fallback`: `true` or `false`

What it measures:

This metric counts individual recommendation items returned by the API. If a
request asks for `k=10`, this counter increases by 10 for that request, assuming
10 recommendations are returned.

Useful queries:

```promql
sum(rate(movie_recommender_recommendations_returned_total[5m])) by (fallback)
```

This shows recommendation item throughput.

How to interpret:

- This is different from request rate. One request can return multiple
  recommendation items.
- If prediction requests are steady but returned items drop, the model artifact
  may not contain enough recommendations for some users.
- Compare this with `movie_recommender_predictions_total` to estimate the
  average number of items returned per request.

Example average items per request:

```promql
sum(rate(movie_recommender_recommendations_returned_total[5m]))
/
sum(rate(movie_recommender_predictions_total[5m]))
```

## Drift Metrics

The current drift monitoring is lightweight serving-time drift monitoring. It
does not compare against a stored statistical baseline yet. Instead, it exposes
signals that show whether the shape of served predictions is changing.

### `movie_recommender_prediction_score`

Type: histogram

Labels:

- `fallback`: `true` or `false`
- `le`: histogram bucket upper bound

Buckets:

```text
0, 1, 2, 3, 4, 5, 10, 20, 50, 100, +Inf
```

What it measures:

This metric records the score of every recommendation item returned by the API.
Personalized ALS recommendations and popularity fallback recommendations may
use different score scales, so the `fallback` label is important.

Useful queries:

```promql
histogram_quantile(
  0.50,
  sum(rate(movie_recommender_prediction_score_bucket[5m])) by (le, fallback)
)
```

This shows median served recommendation score by fallback mode.

```promql
histogram_quantile(
  0.95,
  sum(rate(movie_recommender_prediction_score_bucket[5m])) by (le, fallback)
)
```

This shows p95 served recommendation score by fallback mode.

How to interpret:

- A stable score distribution usually means the model is serving a similar type
  of recommendation over time.
- A sudden score drop may mean lower-confidence recommendations are being
  served.
- A sudden score increase can also be suspicious if it happens after model or
  data changes, because it may indicate a scoring-scale change rather than a
  true quality improvement.
- Compare `fallback=true` and `fallback=false` separately because popularity
  scores and ALS predicted ratings may not be directly comparable.

### `movie_recommender_prediction_release_year_total`

Type: counter

Labels:

- `fallback`: `true` or `false`
- `year_bucket`: release year group

Release year buckets:

```text
unknown
pre_1980
1980s
1990s
2000s
2010s
2020s
```

What it measures:

This metric counts the release-year bucket of every recommendation item returned
by the API.

Useful query:

```promql
sum(increase(movie_recommender_prediction_release_year_total[1h])) by (year_bucket, fallback)
```

This shows the mix of recommended movie eras over the last hour.

How to interpret:

- A stable release-year mix means the API is recommending a similar era
  distribution over time.
- A sudden shift toward old or new movies may indicate model drift, traffic
  drift, fallback behavior changes, or changed model artifacts.
- A high `unknown` bucket means release-year extraction may be incomplete.

## Dashboard Panels

### API request rate

Query:

```promql
sum(rate(movie_recommender_api_requests_total[5m])) by (endpoint, status)
```

Shows API traffic per second by endpoint and status.

Use this to verify:

- the API is receiving traffic
- health and readiness checks are running
- recommendation calls are successful
- `not_ready` or missing-artifact statuses are not increasing

### p95 request latency

Query:

```promql
histogram_quantile(
  0.95,
  sum(rate(movie_recommender_api_request_latency_seconds_bucket[5m])) by (le, endpoint)
)
```

Shows the 95th percentile request latency in seconds.

Use this to verify:

- recommendation latency is acceptable
- latency does not rise during load
- the API remains responsive after model artifact updates

### Prediction request rate

Query:

```promql
sum(rate(movie_recommender_predictions_total[5m])) by (fallback)
```

Shows successful recommendation request rate split by fallback mode.

Use this to verify:

- recommendation traffic is being served
- personalized and fallback traffic are visible separately
- unknown-user traffic is not unexpectedly dominating

### Drift signal: fallback ratio

Query:

```promql
100 * sum(rate(movie_recommender_predictions_total{fallback="true"}[5m]))
/
sum(rate(movie_recommender_predictions_total[5m]))
```

Shows the percentage of recommendation requests using fallback.

Use this to detect:

- user population drift
- missing personalized recommendations
- model artifacts that do not cover current users
- demos accidentally hitting mostly unknown user IDs

Expected behavior:

- Lower is better for personalized recommendation coverage.
- Some fallback is normal for unseen users.
- A sudden sustained increase should be investigated.

### Recommendations returned rate

Query:

```promql
sum(rate(movie_recommender_recommendations_returned_total[5m])) by (fallback)
```

Shows how many recommendation items are returned per second.

Use this to verify:

- the API returns the expected number of items
- fallback and personalized outputs are both producing recommendation lists
- returned item volume matches request volume and requested `k`

### Drift signal: recommendation score distribution

Queries:

```promql
histogram_quantile(
  0.50,
  sum(rate(movie_recommender_prediction_score_bucket[5m])) by (le, fallback)
)
```

```promql
histogram_quantile(
  0.95,
  sum(rate(movie_recommender_prediction_score_bucket[5m])) by (le, fallback)
)
```

Shows median and p95 recommendation score by fallback mode.

Use this to detect:

- score distribution shifts
- scoring-scale changes
- lower-confidence predictions
- unusually high fallback popularity scores

### Drift signal: recommended movie release-year mix

Query:

```promql
sum(increase(movie_recommender_prediction_release_year_total[1h])) by (year_bucket, fallback)
```

Shows how many recommended movies came from each release-year bucket during the
last hour.

Use this to detect:

- recommendation content drifting toward older or newer movies
- unexpected increase in missing release years
- different era distribution between personalized and fallback recommendations

### Prediction score histogram buckets

Query:

```promql
sum(rate(movie_recommender_prediction_score_bucket[5m])) by (le, fallback)
```

Shows the raw score histogram bucket rates.

Use this to debug:

- why score percentile panels look the way they do
- whether scores are clustered in a small bucket range
- whether fallback and personalized scores have different scales

## Why Panels Sometimes Show No Data

Most dashboard panels use `rate(...[5m])`. For `rate()` to produce useful data,
Prometheus needs recent metric samples and at least two scrapes in the selected
range.

Common causes:

- no recent recommendation requests
- API container restarted and counters reset
- Prometheus has not scraped the API yet
- dashboard time range is too narrow
- datasource name or UID does not match the provisioned datasource

Generate demo traffic:

```bash
for i in $(seq 1 30); do
  curl "http://localhost:8000/recommendations/1?k=5" >/dev/null
  curl "http://localhost:8000/recommendations/99999?k=5" >/dev/null
  sleep 1
done
```

Check raw metrics:

```bash
curl http://localhost:8000/metrics | grep movie_recommender
```

Check Prometheus target health:

```bash
curl http://localhost:9090/api/v1/targets
```

## Demo Checklist

1. Start the local stack:

   ```bash
   docker compose up --build
   ```

2. Confirm API readiness:

   ```bash
   curl http://localhost:8000/ready
   ```

3. Generate recommendation traffic:

   ```bash
   for i in $(seq 1 30); do
     curl "http://localhost:8000/recommendations/1?k=5" >/dev/null
     curl "http://localhost:8000/recommendations/99999?k=5" >/dev/null
     sleep 1
   done
   ```

4. Open Grafana:

   ```text
   http://localhost:3000/d/efvbt8hwrwruoe/basic-api-service-dashboard
   ```

5. Explain the main story:

   - API traffic and latency show service health.
   - Prediction rate shows recommendation traffic.
   - Fallback ratio shows whether unknown users are increasing.
   - Score distribution shows whether prediction scores are shifting.
   - Release-year mix shows whether recommended content is drifting.

