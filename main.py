import functools
import http
import mimetypes
import os
import re
import subprocess
import unicodedata
from html import escape
from pathlib import Path
from random import choice
from random import random

import openai
import yaml
from flask import Flask
from flask import request
from fuzzywuzzy import fuzz
from google.cloud.vision import Image
from google.cloud.vision import ImageAnnotatorClient
from telegram import Bot
from telegram import ParseMode
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import CallbackContext
from telegram.ext import CommandHandler
from telegram.ext import Dispatcher
from telegram.ext import Filters
from telegram.ext import MessageHandler
from werkzeug.wrappers import Response

app = Flask(__name__)

vision = ImageAnnotatorClient()

openai.api_key = os.getenv("OPENAI_API_KEY")

mimetypes.init()


with open("memes.yaml", "rt", encoding="utf-8") as f:
    memes = yaml.safe_load(f)

fortunes = memes["fortunes"]
replies = memes["replies"]
slaps = memes["slaps"]
welcome = memes["welcome"]
helps = memes["helps"]
porn = memes["porn"]
likelihoods = tuple(memes["likelihoods"])


def likely(likelihood):
    return likelihood in ["LIKELY", "VERY_LIKELY"]


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
    author = message.from_user.username

    if not reply_to:
        return

    text = message.text.strip()

    if not text:
        return

    success = False

    shell = ["sed", "-r", text]

    result = subprocess.run(shell, text=True, input=reply_to.text, capture_output=True)

    if result.returncode == 0:
        reply = result.stdout.strip()

        if reply:
            try:
                message.delete()
            except TelegramError:
                pass

            success = True
            reply = escape(reply)
            html = f'<b>Você quis dizer:</b>\n"{reply}"'
            reply_to.reply_text(html, parse_mode=ParseMode.HTML)

    if not success:
        reply = f"Ihhh... @{author} não sabe /regex/! 😂"
        context.bot.send_message(chat_id=update.effective_chat.id, text=reply)


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
            image = Image(content=bytes(buffer))

            adult = likelihoods[
                vision.safe_search_detection(image).safe_search_annotation.adult
            ]

            if likely(adult):
                update.message.reply_text(choice(porn))
                break

            annotations = vision.label_detection(image).label_annotations
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


def tramp(update: Update, context: CallbackContext) -> None:
    message = update.message

    if not message:
        return

    reply_to_message = update.message.reply_to_message

    if not reply_to_message:
        return

    with open(Path(r"assets/to get by/0.mp4"), "rb") as f:
        reply_to_message.reply_video(f)

    try:
        message.delete()
    except TelegramError:
        pass


def prompt(update: Update, context: CallbackContext) -> None:
    prompt = re.sub(r"^/prompt(@delduca_bot)?$", "", update.message.text)

    if not prompt:
        return

    update.message.reply_text(
        openai.Completion.create(
            prompt=prompt,
            model="text-davinci-003",
            best_of=3,
            max_tokens=2048,
        )
        .choices[0]
        .text
    )


bot = Bot(token=os.environ["TELEGRAM_TOKEN"])

dispatcher = Dispatcher(bot=bot, update_queue=None)
dispatcher.add_handler(MessageHandler(Filters.regex(r"^s/"), sed))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, meme))
dispatcher.add_handler(MessageHandler(Filters.status_update.new_chat_members, enter))
dispatcher.add_handler(CommandHandler("fortune", fortune))
dispatcher.add_handler(CommandHandler("repost", repost))
dispatcher.add_handler(CommandHandler("rules", rules))
dispatcher.add_handler(CommandHandler("slap", slap))
dispatcher.add_handler(CommandHandler("vagabundo", tramp))
dispatcher.add_handler(CommandHandler("prompt", prompt))


@app.post("/")
def index() -> Response:
    dispatcher.process_update(Update.de_json(request.get_json(force=True), bot))

    return "", http.HTTPStatus.NO_CONTENT
