FROM runpod/base:1.0.3-cuda1290-ubuntu2204

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY runpod/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY runpod/app/ .

ENV EMBED_MODEL=paraphrase-multilingual-mpnet-base-v2

CMD ["python", "handler.py"]