# Base Image
FROM python:3.9-slim

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install make \
    && apt-get -y install espeak-ng

WORKDIR /app
COPY . .

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

# Expose the port the app runs on
EXPOSE 5011