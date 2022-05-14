import functools
import http
import mimetypes
import os
import subprocess
import unicodedata
from html import escape
from pathlib import Path
from random import choice, random

import yaml
from flask import Flask, request
from fuzzywuzzy import fuzz
from google.cloud import vision
from telegram import Bot, ParseMode, Update
from telegram.error import TelegramError
from werkzeug.wrappers import Response
from telegram.ext import (
    CallbackContext,
    CommandHandler,
    Dispatcher,
    Filters,
    MessageHandler,
)

app = Flask(__name__)

vision_client = vision.ImageAnnotatorClient()

mimetypes.init()


with open("memes.yaml", "rt", encoding="utf-8") as f:
    memes = yaml.safe_load(f)

fortunes = memes["fortunes"]
replies = memes["replies"]
slaps = memes["slaps"]
welcome = memes["welcome"]
helps = memes["helps"]


def remove_unicode(string: str) -> str:
    funcs = [
        lambda u: unicodedata.normalize("NFD", u),
        lambda s: s.encode("ascii", "ignore"),
        lambda s: s.decode("utf-8"),
        lambda s: s.strip(),
    ]

    return functools.reduce(lambda x, f: f(x), funcs, string)


def sed(update: Update, context: CallbackContext) -> None:
    message = update.message
    reply_to = message.reply_to_message

    if not reply_to:
        return

    text = message.text.strip()

    if not text:
        return

    shell = ["sed", "-r", text]

    result = subprocess.run(shell, text=True, input=reply_to.text, capture_output=True)

    if result.returncode == 0:
        reply = result.stdout.strip()

        if reply:
            reply = escape(reply)
            html = f'<b>Você quis dizer:</b>\n"{reply}"'
            reply_to.reply_text(html, parse_mode=ParseMode.HTML)

    try:
        message.delete()
    except TelegramError:
        pass


def meme(update: Update, context: CallbackContext) -> None:
    message = update.message

    if not message:
        return

    text = remove_unicode(message.text.lower())

    if not text:
        return

    reply = next((replies[key] for key in text.split() if key in replies), None)

    if reply:
        if random() < 0.1:
            message.reply_text(choice(reply))
            return

    if max([fuzz.ratio(h, text) for h in helps]) > 70:
        filename = Path(r"assets/to get by/0.mp4")

        with open(filename, "rb") as f:
            message.reply_video(f)
            return


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

dispatcher = Dispatcher(bot=bot, update_queue=None)
dispatcher.add_handler(MessageHandler(Filters.regex(r"^s/"), sed))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, meme))
dispatcher.add_handler(MessageHandler(Filters.status_update.new_chat_members, enter))
dispatcher.add_handler(CommandHandler("fortune", fortune))
dispatcher.add_handler(CommandHandler("repost", repost))
dispatcher.add_handler(CommandHandler("rules", rules))
dispatcher.add_handler(CommandHandler("slap", slap))


@app.post("/")
def index() -> Response:
    dispatcher.process_update(Update.de_json(request.get_json(force=True), bot))

    return "", http.HTTPStatus.NO_CONTENT
