import os
import http

import yaml

from flask import Flask, request
from werkzeug.wrappers import Response

from telegram import Bot, Update
from telegram.ext import Dispatcher, Filters, MessageHandler, CallbackContext

from google.cloud import vision

app = Flask(__name__)

vision_client = vision.ImageAnnotatorClient()


with open("memes.yaml") as f:
    memes = yaml.load(f, Loader=yaml.FullLoader)


def memify(update: Update, context: CallbackContext) -> None:
    # update.message.reply_text(update.message.text)
    pass


def add_group(update: Update, context: CallbackContext) -> None:
    for member in update.message.new_chat_members:
        photos = member.get_profile_photos().photos
        for photo in photos:
            photo_file = context.bot.getFile(photo[-1].file_id)
            buffer = photo_file.download_as_bytearray()
            image = vision.Image(content=bytes(buffer))
            response = vision_client.label_detection(image=image)
            annotations = response.label_annotations
            labels = set([label.description.lower() for label in annotations])
            if "anime" in labels:
                update.message.reply_text("Um otaku entrou no grupo!")
                break


bot = Bot(token=os.environ["TOKEN"])

dispatcher = Dispatcher(bot=bot, update_queue=None, workers=0)
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, memify))
dispatcher.add_handler(MessageHandler(Filters.status_update.new_chat_members, add_group))


@app.route("/", methods=["POST"])
def index() -> Response:
    dispatcher.process_update(
        Update.de_json(request.get_json(force=True), bot))

    return "", http.HTTPStatus.NO_CONTENT
