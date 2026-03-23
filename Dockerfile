FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharger le modèle au build — pas au cold start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"

COPY app/ .

ENV EMBED_MODEL=paraphrase-multilingual-mpnet-base-v2

CMD ["python", "handler.py"]