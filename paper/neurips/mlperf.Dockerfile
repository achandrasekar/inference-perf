FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy loadgen source and build it
COPY loadgen /app/loadgen
RUN cd /app/loadgen && pip install .

# Copy client script
COPY mlperf_client.py /app/mlperf_client.py

RUN pip install aiohttp numpy

ENTRYPOINT ["python", "/app/mlperf_client.py"]
