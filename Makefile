.PHONY: help deploy vet update

.SILENT:

help:
	awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

deploy: ## Deploy on Google Cloud Run
	gcloud config set run/region us-central1
	gcloud run deploy delduca --source $(shell pwd) --platform managed --allow-unauthenticated --project bots-for-telegram

vet: ## Run linters, auto-formaters, and other tools
	black main.py
	flake8 --max-line-length=88 main.py
	isort --force-single-line-imports main.py
