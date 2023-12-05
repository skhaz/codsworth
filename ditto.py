import asyncio
import io

import aiofiles
import jinja2
import numpy as np
import PIL.Image
from playwright.async_api import async_playwright


def trim(buffer, margin=32, color="#92b88a"):
    with PIL.Image.open(io.BytesIO(buffer)).convert("RGB") as image:
        data = np.array(image)
        trim = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))

        non_trim_rows = np.where(np.any(data != trim, axis=(1, 2)))[0]
        non_trim_cols = np.where(np.any(data != trim, axis=(0, 2)))[0]

        left = max(0, non_trim_cols[0] - margin)
        top = max(0, non_trim_rows[0] - margin)
        right = min(data.shape[1] - 1, non_trim_cols[-1] + margin)
        bottom = min(data.shape[0] - 1, non_trim_rows[-1] + margin)

        return PIL.Image.fromarray(data[top:bottom, left:right])


async def ditto() -> bytes | None:
    async with (
        aiofiles.open("assets/ditto/index.html", "rt") as html,
        async_playwright() as playwright,
    ):
        chromium = playwright.chromium

        browser = await chromium.launch(
            headless=True,
            args=[
                "--autoplay-policy=user-gesture-required",
                "--disable-background-networking",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-breakpad",
                "--disable-client-side-phishing-detection",
                "--disable-component-update",
                "--disable-default-apps",
                "--disable-dev-shm-usage",
                "--disable-domain-reliability",
                "--disable-extensions",
                "--disable-features=AudioServiceOutOfProcess",
                "--disable-hang-monitor",
                "--disable-ipc-flooding-protection",
                "--disable-notifications",
                "--disable-offer-store-unmasked-wallet-cards",
                "--disable-popup-blocking",
                "--disable-print-preview",
                "--disable-prompt-on-repost",
                "--disable-renderer-backgrounding",
                "--disable-setuid-sandbox",
                "--disable-speech-api",
                "--disable-sync",
                "--disk-cache-size=33554432",
                "--hide-scrollbars",
                "--ignore-gpu-blacklist",
                "--metrics-recording-only",
                "--mute-audio",
                "--no-default-browser-check",
                "--no-first-run",
                "--no-pings",
                "--no-sandbox",
                "--no-zygote",
                "--password-store=basic",
                "--use-gl=swiftshader",
                "--use-mock-keychain",
                "--single-process",
            ],
        )

        context = await browser.new_context(
            viewport={
                "width": 1600,
                "height": 1200,
            },
            device_scale_factor=2,
        )

        page = await context.new_page()

        context = {
            "messages": [
                {
                    "avatar": "https://github.com/skhaz.png",
                    "name": "Rodrigo",
                    "role": "admin",
                    "content": "Piroca",
                },
                {
                    "avatar": "https://github.com/walac.png",
                    "name": "Wander",
                    "role": "admin",
                    "content": "Finalmente sai da bolsa!",
                },
            ]
        }

        environment = jinja2.Environment()
        template = environment.from_string(await html.read())
        await page.set_content(template.render(context))

        screenshot = await page.screenshot()

        coro = asyncio.to_thread(trim, screenshot)

        try:
            result = await asyncio.wait_for(coro, timeout=30)
        except asyncio.TimeoutError:
            return None

        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return buffer.getvalue()
