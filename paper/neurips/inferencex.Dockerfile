FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/SemiAnalysisAI/InferenceX.git

WORKDIR /workspace/InferenceX

RUN pip install --no-cache-dir transformers numpy tqdm aiohttp

ENV PYTHONPATH=/workspace/InferenceX:/workspace/InferenceX/utils/bench_serving:$PYTHONPATH

ENTRYPOINT ["python", "utils/bench_serving/benchmark_serving.py"]
