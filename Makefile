.PHONY: help generate train up demo down logs test clean

help:
	@echo "make generate  - create synthetic transaction dataset (CSV)"
	@echo "make train     - train XGBoost model + register in MLflow"
	@echo "make up        - start full platform (kafka, redis, postgres, mlflow, inference, stream, dashboard)"
	@echo "make demo      - start transaction replay producer (kafka stream)"
	@echo "make logs      - tail all service logs"
	@echo "make test      - run unit tests"
	@echo "make down      - stop everything and delete volumes"

generate:
	python -m src.datagen.synth_generator --rows 20000 --out data/processed/demo_transactions.csv

train:
	python -m src.training.train --csv data/processed/demo_transactions.csv

up:
	docker compose up -d --build

demo:
	docker compose --profile demo up -d producer

logs:
	docker compose logs -f

test:
	pytest -q

down:
	docker compose --profile demo down -v

clean:
	rm -rf artifacts/models/* data/processed/* mlruns
