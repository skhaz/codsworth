
Deploy

```bash
export PROJECT_ID=telegram-delduca-bot
export TOKEN=your-telegram-bot-token
```

```bash
gcloud run deploy bot \
    --source $(pwd) \
    --region us-central1 \
    --allow-unauthenticated \
    --platform managed \
    --project ${PROJECT_ID}
```

Set Webhook (only need to be done once)

```shell
curl "https://api.telegram.org/bot${TOKEN}/setWebhook?url=$(gcloud run services describe bot --format 'value(status.url)' --project ${PROJECT_ID})"
```
