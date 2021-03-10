import os
import http
import subprocess
import mimetypes

from pathlib import Path
from html import escape
from random import random, choice

import yaml

from flask import Flask, request
from werkzeug.wrappers import Response

from telegram import Bot, Update, ParseMode
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

mimetypes.init()


with open("memes.yaml") as f:
    memes = yaml.load(f, Loader=yaml.FullLoader)

fortunes = memes["fortunes"]
replies = memes["replies"]
slaps = memes["slaps"]
welcome = memes["welcome"]


def sed(update: Update, context: CallbackContext) -> None:
    message = update.message
    reply_to = message.reply_to_message
    if not reply_to:
        return
    result = subprocess.run(["sed", "-r", message.text], text=True, input=reply_to.text, capture_output=True)
    if result.returncode == 0:
        reply = result.stdout.strip()
        if reply:
            reply = escape(reply)
            html = f'<b>Você quis dizer:</b>\n"{reply}"'
            reply_to.reply_text(html, parse_mode=ParseMode.HTML)
    try:
        message.delete()
    except:
        pass


def meme(update: Update, context: CallbackContext) -> None:
    message = update.message
    if not message:
        return
    keywords = message.text.lower().split()
    reply = next((replies[key] for key in keywords if key in replies), None)
    if reply:
        if random() < 0.2:
            message.reply_text(choice(reply))


def enter(update: Update, context: CallbackContext) -> None:
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
    message.reply_text(choice(fortunes))


def repost(update: Update, context: CallbackContext) -> None:
    message = update.message.reply_to_message
    if not message:
        return
    assets = Path("assets/repost")
    filename = choice(list(assets.iterdir()))
    mimetype, _ = mimetypes.guess_type(filename)
    reply_with = getattr(message, f"reply_{mimetype.split('/')[0]}")
    with open(filename, "rb") as f:
        reply_with(f)


def rules(update: Update, context: CallbackContext) -> None:
    message = update.message.reply_to_message or update.message
    if not message:
        return
    assets = Path("assets/rules")
    filename = choice(list(assets.iterdir()))
    with open(filename, "rb") as f:
        message.reply_video(f)


def slap(update: Update, context: CallbackContext) -> None:
    message = update.message.reply_to_message
    if message:
        message.reply_text(choice(slaps))


bot = Bot(token=os.environ["TOKEN"])

dispatcher = Dispatcher(bot=bot, update_queue=None, workers=0)
dispatcher.add_handler(MessageHandler(Filters.regex(r"^s/"), sed))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, meme))
dispatcher.add_handler(MessageHandler(Filters.status_update.new_chat_members, enter))
dispatcher.add_handler(CommandHandler("fortune", fortune))
dispatcher.add_handler(CommandHandler("repost", repost))
dispatcher.add_handler(CommandHandler("rules", rules))
dispatcher.add_handler(CommandHandler("slap", slap))


@app.route("/", methods=["POST"])
def index() -> Response:
    dispatcher.process_update(
        Update.de_json(request.get_json(force=True), bot))

    return "", http.HTTPStatus.NO_CONTENT
