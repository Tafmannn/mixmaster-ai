FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg git && apt-get clean

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "predict.py"]