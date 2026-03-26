FROM runpod/base:0.6.2-cpu

RUN apt-get update && apt-get install -y git python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY runpod/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY runpod/app/ .

ENV EMBED_MODEL=paraphrase-multilingual-mpnet-base-v2

CMD ["python", "handler.py"]