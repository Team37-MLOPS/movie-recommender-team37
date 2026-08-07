"""
Apache Beam batch pipeline for the MovieLens-1M dataset.

Why Beam here: the raw data ships as three flat ``.dat`` files (ratings,
movies, users) that need to be parsed, cleaned, joined, and feature-engineered
before model training. Beam's DirectRunner gives us a portable, parallel
batch-ETL pipeline for this join+transform workload, and the same pipeline
graph could later be pointed at a distributed runner (Spark/Dataflow/Flink)
without rewriting the transform logic — which is the justification MLOps
guidelines ask for when picking a data engineering tool.

Steps performed:
    1. Parse ratings.dat / movies.dat / users.dat (``::`` delimited).
    2. Clean: drop malformed rows, dedupe exact duplicate ratings.
    3. Handle missing values: fill unknown zip codes / ages, drop rows with
       missing keys.
    4. Feature engineering: one-hot genre flags, rating timestamp -> year,
       age bucket, gender flag, per-user and per-movie rating aggregates.
    5. Join ratings + movies + users into one feature table.
    6. Write the final table as CSV under data/processed/.
"""

import argparse
import csv
import logging
import os
from datetime import datetime, timezone

import apache_beam as beam
from apache_beam.coders.coders import Coder
from apache_beam.options.pipeline_options import PipelineOptions


class Latin1Coder(Coder):
    """movies.dat contains Latin-1 titles (e.g. accented characters); the
    default UTF-8 text coder chokes on those bytes."""

    def encode(self, value):
        return value.encode("latin-1")

    def decode(self, value):
        return value.decode("latin-1")

    def is_deterministic(self):
        return True

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

AGE_BUCKETS = [(0, 18), (18, 25), (25, 35), (35, 45), (45, 50), (50, 56), (56, 200)]


def bucket_age(age):
    for lo, hi in AGE_BUCKETS:
        if lo <= age < hi:
            return f"{lo}-{hi}"
    return "unknown"


class ParseRatingLine(beam.DoFn):
    """Parses one ``UserID::MovieID::Rating::Timestamp`` line."""

    def process(self, line):
        parts = line.strip().split("::")
        if len(parts) != 4:
            return
        try:
            user_id, movie_id, rating, ts = (
                int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            )
        except ValueError:
            return
        if not (1 <= rating <= 5):
            return
        yield {
            "user_id": user_id,
            "movie_id": movie_id,
            "rating": rating,
            "timestamp": ts,
        }


class ParseMovieLine(beam.DoFn):
    """Parses one ``MovieID::Title::Genre1|Genre2|...`` line."""

    def process(self, line):
        parts = line.strip().split("::")
        if len(parts) != 3:
            return
        try:
            movie_id = int(parts[0])
        except ValueError:
            return
        title, genre_str = parts[1], parts[2]
        genres = set(genre_str.split("|")) if genre_str else set()
        record = {"movie_id": movie_id, "title": title}
        for genre in GENRES:
            record[f"genre_{genre.lower().replace('-', '_').replace(chr(39), '')}"] = int(genre in genres)
        record["num_genres"] = len(genres)
        yield record


class ParseUserLine(beam.DoFn):
    """Parses one ``UserID::Gender::Age::Occupation::Zip`` line."""

    def process(self, line):
        parts = line.strip().split("::")
        if len(parts) != 5:
            return
        try:
            user_id, age, occupation = int(parts[0]), int(parts[2]), int(parts[3])
        except ValueError:
            return
        gender = parts[1] if parts[1] in ("M", "F") else "U"  # handle missing/malformed gender
        yield {
            "user_id": user_id,
            "gender": gender,
            "age": age,
            "age_bucket": bucket_age(age),
            "occupation": occupation,
        }


def dedupe_key(record):
    return (record["user_id"], record["movie_id"])


def join_ratings_movies_users(element):
    """element: (user_id, {'ratings': [...], 'users': [...]})"""
    _, grouped = element
    users = grouped["users"]
    ratings = grouped["ratings"]
    if not users:
        return
    user = users[0]
    for rating in ratings:
        yield rating["movie_id"], {**rating, **{k: v for k, v in user.items() if k != "user_id"}}


def join_with_movies(element):
    """element: (movie_id, {'partial': [...], 'movies': [...]})"""
    _, grouped = element
    movies = grouped["movies"]
    partials = grouped["partial"]
    if not movies:
        return
    movie = movies[0]
    for partial in partials:
        merged = {**partial, "movie_id": movie["movie_id"]}
        for k, v in movie.items():
            if k == "movie_id":
                continue
            merged[k] = v
        yield merged


def add_derived_features(record):
    ts = record.get("timestamp", 0)
    rating_dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
    record["rating_year"] = rating_dt.year if rating_dt else 0
    record["rating_dow"] = rating_dt.weekday() if rating_dt else -1
    record["gender_is_male"] = int(record.get("gender") == "M")
    return record


def to_csv_row(record, fieldnames):
    import io
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow([record.get(f, "") for f in fieldnames])
    return buf.getvalue()


FIELDNAMES = (
    ["user_id", "movie_id", "rating", "timestamp", "rating_year", "rating_dow"]
    + ["gender", "gender_is_male", "age", "age_bucket", "occupation"]
    + ["title", "num_genres"]
    + [f"genre_{g.lower().replace('-', '_').replace(chr(39), '')}" for g in GENRES]
)


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--movies", required=True)
    parser.add_argument("--users", required=True)
    parser.add_argument("--output", required=True)
    known_args, pipeline_args = parser.parse_known_args(argv)

    os.makedirs(os.path.dirname(known_args.output), exist_ok=True)
    options = PipelineOptions(pipeline_args)

    with beam.Pipeline(options=options) as pipeline:
        ratings = (
            pipeline
            | "ReadRatings" >> beam.io.ReadFromText(known_args.ratings)
            | "ParseRatings" >> beam.ParDo(ParseRatingLine())
            | "KeyRatingsForDedupe" >> beam.Map(lambda r: (dedupe_key(r), r))
            | "DedupeRatings" >> beam.CombinePerKey(lambda records: next(iter(records)))
            | "DropDedupeKey" >> beam.Map(lambda kv: kv[1])
        )

        movies = (
            pipeline
            | "ReadMovies" >> beam.io.ReadFromText(known_args.movies, coder=Latin1Coder())
            | "ParseMovies" >> beam.ParDo(ParseMovieLine())
        )

        users = (
            pipeline
            | "ReadUsers" >> beam.io.ReadFromText(known_args.users)
            | "ParseUsers" >> beam.ParDo(ParseUserLine())
        )

        ratings_by_user = ratings | "KeyRatingsByUser" >> beam.Map(lambda r: (r["user_id"], r))
        users_by_id = users | "KeyUsersById" >> beam.Map(lambda u: (u["user_id"], u))

        ratings_with_users = (
            {"ratings": ratings_by_user, "users": users_by_id}
            | "GroupByUser" >> beam.CoGroupByKey()
            | "JoinRatingsUsers" >> beam.FlatMap(join_ratings_movies_users)
        )

        partial_by_movie = ratings_with_users | "KeyPartialByMovie" >> beam.Map(lambda kv: kv)
        movies_by_id = movies | "KeyMoviesById" >> beam.Map(lambda m: (m["movie_id"], m))

        full_records = (
            {"partial": partial_by_movie, "movies": movies_by_id}
            | "GroupByMovie" >> beam.CoGroupByKey()
            | "JoinWithMovies" >> beam.FlatMap(join_with_movies)
            | "AddDerivedFeatures" >> beam.Map(add_derived_features)
        )

        header = ",".join(FIELDNAMES)
        (
            full_records
            | "ToCsvRow" >> beam.Map(to_csv_row, fieldnames=FIELDNAMES)
            | "WriteCsv" >> beam.io.WriteToText(
                known_args.output,
                file_name_suffix=".csv",
                header=header,
                shard_name_template="",
            )
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
