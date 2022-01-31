
Deploy

```bash
export PROJECT_ID=bots-for-telegram
export TOKEN=your-telegram-bot-token
```

```bash
gcloud run deploy delduca \
    --source $(pwd) \
    --region us-central1 \
    --allow-unauthenticated \
    --platform managed \
    --set-env-vars TOKEN=${TOKEN} \
    --project ${PROJECT_ID}
```

Set Webhook (only need to be done once)

```shell
curl "https://api.telegram.org/bot${TOKEN}/setWebhook?url=$(gcloud run services describe delduca --format 'value(status.url)' --project ${PROJECT_ID})"
```
