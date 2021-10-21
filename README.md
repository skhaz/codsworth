
Deploy

```bash
gcloud run deploy bot \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --platform managed \
    --project telegram-delduca-bot
```