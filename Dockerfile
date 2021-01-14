FROM python:3.8-slim

ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY *.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app