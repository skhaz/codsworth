import functools
import mimetypes
import os
import re
import subprocess
import unicodedata
from functools import wraps
from html import escape
from http import HTTPStatus
from pathlib import Path
from queue import Queue
from random import choice
from random import random
from typing import Union

import openai
import sentry_sdk
import yaml
from flask import Flask
from flask import make_response
from flask import request
from fuzzywuzzy import fuzz
from google.cloud.vision import Image
from google.cloud.vision import ImageAnnotatorClient
from redis import ConnectionPool
from redis import Redis
from redis_rate_limit import RateLimit
from redis_rate_limit import TooManyRequests
from telegram import Bot
from telegram import ChatAction
from telegram import ParseMode
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import CallbackContext
from telegram.ext import CommandHandler
from telegram.ext import Dispatcher
from telegram.ext import Filters
from telegram.ext import MessageHandler
from telegram.utils.helpers import escape_markdown
from werkzeug.wrappers import Response

app = Flask(__name__)

vision = ImageAnnotatorClient()

redis_pool = ConnectionPool.from_url(os.environ["REDIS_DSN"])
redis = Redis(connection_pool=redis_pool)

openai.api_key = os.environ["OPENAI_API_KEY"]

mimetypes.init()

sentry_sdk.init(dsn=os.environ["SENTRY_DSN"])

with open("memes.yaml", "rt", encoding="utf-8") as f:
    memes = yaml.safe_load(f)

fortunes = memes["fortunes"]
replies = memes["replies"]
slaps = memes["slaps"]
welcome = memes["welcome"]
helps = memes["helps"]
penis = memes["penis"]
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


def mention_html(user_id: Union[int, str], name: str) -> str:
    return f'<a href="tg://user?id={user_id}">{escape(name)}</a>'


def typing(func):
    @wraps(func)
    def wrapper(update, context, *args, **kwargs):
        context.bot.send_chat_action(
            chat_id=update.effective_message.chat_id,
            action=ChatAction.TYPING,
        )

        return func(update, context, *args, **kwargs)

    return wrapper


@typing
def sed(update: Update, context: CallbackContext) -> None:
    message = update.message
    reply_to = message.reply_to_message

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
        chat = update.effective_chat

        if not chat:
            return

        user = message.from_user
        mention = mention_html(user_id=user.id, name=user.name)
        reply = f"Ihhh... {mention} não sabe /regex/! 😂"
        context.bot.send_message(chat_id=chat.id, text=reply, parse_mode=ParseMode.HTML)


def meme(update: Update, context: CallbackContext) -> None:
    message = update.message

    if not message:
        return

    text = message.text

    if not text:
        return

    escaped_text = escape_markdown(text, version=2)
    length = len(escaped_text)
    indexes = []
    for char in penis:
        begin = indexes[-1] if indexes else 0
        index = escaped_text[begin:length].lower().find(char)
        if index != -1:
            indexes.append(index + begin)

    in_sequence = any(indexes[i] + 1 == indexes[i + 1] for i in range(len(indexes) - 1))
    is_equidistant = len(indexes) == len(penis)
    if is_equidistant and not in_sequence:
        letters = [char for char in escaped_text]
        for i, index in enumerate(indexes):
            letters.insert(index + i, "*")

        indexes = [i for i, char in enumerate(letters) if char == "*"]
        for i, index in enumerate(indexes):
            letters.insert((index + 2) + i, "*")

        user_id = message.from_user.id
        user = message.from_user.name

        pipeline = redis.pipeline(transaction=False)
        pipeline.incr(penis)
        pipeline.incr(f"{penis}:count:{user_id}")
        pipeline.set(f"{penis}:user:{user_id}", user)
        count, count_by_author, _ = pipeline.execute()

        caption = [
            f"Hidden penis detected! {count} penises have been discovered so far. "
            f"{user} has already worshiped the {penis} {count_by_author} time(s).",
        ]

        messages = [
            "".join(letters),
            escape_markdown("".join(caption), version=2),
        ]

        message.reply_text("\n\n".join(messages), parse_mode=ParseMode.MARKDOWN_V2)
        return

    text = remove_unicode(text.lower())

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


def on_enter(update: Update, context: CallbackContext) -> None:
    for member in update.message.new_chat_members:
        profile_photos = member.get_profile_photos()

        if not profile_photos:
            continue

        for photo in profile_photos.photos:
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


@typing
def fortune(update: Update, context: CallbackContext) -> None:
    message = update.message.reply_to_message or update.message

    if not message:
        return

    message.reply_text(choice(fortunes))


def repost(update: Update, context: CallbackContext) -> None:
    message = update.message.reply_to_message

    if not message:
        return

    assets = Path("assets/repost")
    filename = choice(list(assets.iterdir()))
    mimetype, _ = mimetypes.guess_type(filename)

    if not mimetype:
        return

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

    reply_to_message = message.reply_to_message

    if not reply_to_message:
        return

    try:
        message.delete()
    except TelegramError:
        pass

    with open(Path(r"assets/to get by/0.mp4"), "rb") as f:
        reply_to_message.reply_video(f)


def ban(update: Update, context: CallbackContext) -> None:
    message = update.message

    if not message:
        return

    reply_to = message.reply_to_message

    if not reply_to:
        return

    user_id = reply_to.from_user.id
    user = reply_to.from_user.name
    pipeline = redis.pipeline(transaction=False)
    pipeline.incr(f"ban:count:{user_id}")
    pipeline.set(f"ban:user:{user_id}", user)
    times = pipeline.execute()[0]
    mention = mention_html(user_id=user_id, name=user)
    text = f"{mention} já foi banido {times} vez(es) 👮‍♂️"
    context.bot.send_message(message.chat_id, text, parse_mode=ParseMode.HTML)


def leaderboard(update: Update, context: CallbackContext) -> None:
    message = update.message

    if not message:
        return

    get_count = (
        lambda key: int(value.decode()) if (value := redis.get(key)) is not None else 0
    )

    pederasts = {
        key.decode().split(":")[2]: get_count(key)
        for key in redis.scan_iter(f"{penis}:count:*")
    }

    sorted_pederasts = sorted(pederasts.items(), key=lambda x: x[1], reverse=True)[:10]

    get_username = (
        lambda key: username.decode()
        if (username := redis.get(key)) is not None
        else None
    )

    arr = [
        rf"\* [{get_username(f'{penis}:user:{pederast[0]}')}](tg://user?id={pederast[0]}) worshipped the {penis} {pederast[1]} times\."
        for pederast in sorted_pederasts  # noqa
    ]

    message.reply_text("\n".join(arr), parse_mode=ParseMode.MARKDOWN_V2)


@typing
def reply(update: Update, context: CallbackContext) -> None:
    message = update.message

    if not message:
        return

    reply_to = message.reply_to_message

    if not reply_to:
        return

    text = reply_to.text

    if not text:
        return

    messages = [
        {
            "role": "system",
            "content": "You are a broadcaster; comment about the following message using the same language, the comment must be ironic or sarcastic, and it must be short and objective.",  # noqa
        },
        {"role": "user", "content": text},
    ]

    try:
        with RateLimit(
            redis_pool=redis_pool,
            resource=message.chat_id,
            client=message.from_user.username,
            max_requests=1,
            expire=60 * 5,
        ):
            reply_to.reply_text(
                openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                )
                .choices[0]
                .message.content
            )
    except TooManyRequests:
        pass


@typing
def prompt(update: Update, context: CallbackContext) -> None:
    message = update.message

    if not message:
        return

    prompt = re.sub(r"^/prompt(@delduca_bot)?$", "", message.text)

    if not prompt:
        return

    try:
        with RateLimit(
            redis_pool=redis_pool,
            resource=message.chat_id,
            client=message.from_user.username,
            max_requests=1,
            expire=60 * 5,
        ):
            message.reply_text(
                openai.Completion.create(
                    prompt=prompt,
                    model="text-davinci-003",
                    max_tokens=3000,
                    best_of=3,
                )
                .choices[0]
                .text
            )
    except TooManyRequests:
        pass


@typing
def image(update: Update, context: CallbackContext) -> None:
    message = update.message

    if not message:
        return

    prompt = re.sub(r"^/image(@delduca_bot)?$", "", message.text)

    if not prompt:
        return

    try:
        with RateLimit(
            redis_pool=redis_pool,
            resource=message.chat_id,
            client=message.from_user.username,
            max_requests=1,
            expire=60 * 5,
        ):
            response = openai.Image.create(prompt=prompt, size="512x512")
            message.reply_photo(photo=response["data"][0]["url"])
    except openai.InvalidRequestError:
        mention = mention_html(
            user_id=message.from_user.id,
            name=message.from_user.name,
        )
        text = f"{mention}, jovem, controle seus hormônios... 🍌"
        message.reply_text(text=text, parse_mode=ParseMode.HTML)
    except TooManyRequests:
        pass


def error_handler(update: object, context: CallbackContext) -> None:
    if not isinstance(update, Update):
        return

    error = context.error

    if not error:
        return

    message = update.effective_message

    if not message:
        return

    mention = mention_html(user_id=message.from_user.id, name=message.from_user.name)
    caption = f"{mention} me causou câncer 💀"
    filename = choice(list(Path("assets/died").iterdir()))
    with open(filename, "rb") as f:
        context.bot.send_photo(
            message.chat_id,
            caption=caption,
            photo=f,
            parse_mode=ParseMode.HTML,
        )

    raise error


bot = Bot(token=os.environ["TELEGRAM_TOKEN"])

"""
fortune - tell someone's fortune
repost - tell them that post already have been posted
rules - tell the rules
slap - slaps someone else
vagabundo - tell the person is lazy
ban - bans someone else
leaderboard - tbd
reply - reply to a message using ChatGPT
prompt - generate a text using AI
image - generate a image using AI
rank - To Do
"""
dispatcher = Dispatcher(bot=bot, update_queue=Queue())
dispatcher.add_handler(MessageHandler(Filters.regex(r"^s/"), sed))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, meme))
dispatcher.add_handler(MessageHandler(Filters.status_update.new_chat_members, on_enter))
dispatcher.add_handler(CommandHandler("fortune", fortune))
dispatcher.add_handler(CommandHandler("repost", repost))
dispatcher.add_handler(CommandHandler("rules", rules))
dispatcher.add_handler(CommandHandler("slap", slap))
dispatcher.add_handler(CommandHandler("vagabundo", tramp))
dispatcher.add_handler(CommandHandler("ban", ban))
dispatcher.add_handler(CommandHandler("leaderboard", leaderboard))
dispatcher.add_handler(CommandHandler("reply", reply))
dispatcher.add_handler(CommandHandler("prompt", prompt))
dispatcher.add_handler(CommandHandler("image", image))
# dispatcher.add_error_handler(error_handler)


@app.post("/")
def index() -> Response:
    dispatcher.process_update(Update.de_json(request.get_json(force=True), bot))

    response = make_response()
    response.status_code = HTTPStatus.NO_CONTENT
    response.mimetype = "application/json"
    return response
