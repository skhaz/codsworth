import os
import http
import random

import yaml

from flask import Flask, request
from werkzeug.wrappers import Response

from telegram import Bot, Update
from telegram.ext import (
    Dispatcher,
    Filters,
    CallbackContext,
    CommandHandler,
    MessageHandler,
)

from google.cloud import vision

app = Flask(__name__)

vision_client = vision.ImageAnnotatorClient()


with open("memes.yaml") as f:
    memes = yaml.load(f, Loader=yaml.FullLoader)

fortunes = memes["fortunes"]
replies = memes["replies"]
slaps = memes["slaps"]
welcome = memes["welcome"]


def memify(update: Update, context: CallbackContext) -> None:
    message = update.message
    if not message:
        return

    text = message.text
    if not text:
        return

    keywords = text.lower().split()

    reply = next((replies[key] for key in keywords if key in replies), None)

    if reply:
        if random.random() < 0.2:
            message.reply_text(random.choice(reply))


def on_enter(update: Update, context: CallbackContext) -> None:
    for member in update.message.new_chat_members:

        photos = member.get_profile_photos().photos

        for photo in photos:
            buffer = context.bot.getFile(photo[-1].file_id).download_as_bytearray()
            image = vision.Image(content=bytes(buffer))
            response = vision_client.label_detection(image=image)
            annotations = response.label_annotations
            labels = set([label.description.lower() for label in annotations])
            message = next((welcome[key] for key in labels if key in welcome), None)

            if message:
                update.message.reply_text(message)
                break


def fortune(update: Update, context: CallbackContext) -> None:
    message = update.message.reply_to_message or update.message
    message.reply_text(random.choice(fortunes))


def slap(update: Update, context: CallbackContext) -> None:
    message = update.message.reply_to_message
    if message:
        message.reply_text(random.choice(slaps))


bot = Bot(token=os.environ["TOKEN"])

dispatcher = Dispatcher(bot=bot, update_queue=None, workers=0)
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, memify))
dispatcher.add_handler(MessageHandler(Filters.status_update.new_chat_members, on_enter))
dispatcher.add_handler(CommandHandler("fortune", fortune))
dispatcher.add_handler(CommandHandler("slap", slap))


@app.route("/", methods=["POST"])
def index() -> Response:
    dispatcher.process_update(
        Update.de_json(request.get_json(force=True), bot))

    return "", http.HTTPStatus.NO_CONTENT
