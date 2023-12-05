FROM python:3.12-slim-bookworm AS base

ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PATH=/opt/venv/bin:$PATH

FROM base AS builder
WORKDIR /opt/venv
RUN python -m venv .
COPY requirements.txt .
RUN pip install --no-cache-dir --requirement requirements.txt

FROM base
WORKDIR /opt/app
COPY --from=builder /opt/venv /opt/venv
COPY . .
RUN playwright install chromium
RUN playwright install-deps chromium

RUN useradd -r user
USER user

CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 8 --timeout-keep-alive 600 --timeout-graceful-shutdown 600