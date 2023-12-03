FROM python:3.10-slim AS base

ENV PATH /opt/venv/bin:$PATH
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

FROM base AS builder
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN pip install --no-cache-dir --requirement requirements.txt

FROM base
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
RUN apt-get update && apt-get install --yes --no-install-recommends sed mime-support libjemalloc2
COPY . .

WORKDIR /opt
ENV PLAYWRIGHT_BROWSERS_PATH /opt/playwright
RUN playwright install chromium
RUN playwright install install-deps

WORKDIR /app
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libjemalloc.so.2
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
