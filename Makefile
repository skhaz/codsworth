.PHONY: deploy format update

.SILENT:

deploy:
	gcloud config set run/region us-central1
	gcloud run deploy delduca --source $(shell pwd) --platform managed --allow-unauthenticated --project bots-for-telegram

format:
	isort --force-single-line-imports main.py
	black main.py

update:
	pur -r requirements.txt