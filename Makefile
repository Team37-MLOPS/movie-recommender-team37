.PHONY: install test lint extract preprocess train api docker-up docker-down

install:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest -q

lint:
	ruff check src api dags tests scripts

extract:
	PYTHONPATH=src:. python -m movie_recommender.data.extract

preprocess:
	PYTHONPATH=src:. python -m movie_recommender.data.preprocess_spark

train:
	PYTHONPATH=src:. python -m movie_recommender.models.train

api:
	PYTHONPATH=src:. uvicorn api.main:app --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down
