from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from ditto import ditto

# from telegram import Update
# from telegram.ext import Application

# application = (
#     Application.builder().token(os.environ["TELEGRAM_TOKEN"]).updater(None).build()
# )


async def test(request: Request):
    image = await ditto()
    return Response(content=image, media_type="image/png")


# def equals(left: str | None, right: str | None) -> bool:
#     if not left or not right:
#         return False

#     if len(left) != len(right):
#         return False

#     for c1, c2 in zip(left, right):
#         if c1 != c2:
#             return False

#     return True


# async def webhook(request: Request):
#     if not equals(
#         request.headers.get("X-Telegram-Bot-Api-Secret-Token"),
#         os.environ["SECRET"],
#     ):
#         return Response(content="Unauthorized", status_code=401)

#     payload = await request.json()

#     async with application:
#         await application.process_update(Update.de_json(payload, application.bot))

#     return Response(status_code=200)


app = Starlette(
    routes=[
        Route("/", test, methods=["GET"]),
        # Route("/", webhook, methods=["POST"]),
    ],
)
