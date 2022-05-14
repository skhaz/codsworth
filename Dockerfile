FROM python:3.10-slim

ENV PYTHONUNBUFFERED True

RUN apt-get update -y && apt-get install -y sed mime-support

WORKDIR /app

COPY *.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

RUN useradd -r user

USER user

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app