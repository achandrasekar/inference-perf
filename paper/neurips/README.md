# Benchmark Tools for inference-perf Comparison

This directory contains Dockerfiles and Kubernetes Job manifests for 5 different benchmark tools to compare with `inference-perf`.

## Tools Included

1. **vLLM Benchmark**: Uses `vllm bench serve`.
2. **Inference X**: Uses `benchmark_serving.py` from SemiAnalysis.
3. **Guide LLM**: Uses `guidellm`.
4. **AI Perf**: Uses `aiperf`.
5. **Inference Perf**: Uses the latest published version of `inference-perf`.

## How to Run

### 1. Build and Load Images

Build the Docker images for each tool:

**From `paper/neurips/` directory:**
```bash
docker build -t vllm-bench:latest -f vllm.Dockerfile .
docker build -t inferencex-bench:latest -f inferencex.Dockerfile .
docker build -t guidellm-bench:latest -f guidellm.Dockerfile .
docker build -t aiperf-bench:latest -f aiperf.Dockerfile .
```

**From the project root directory:**
```bash
docker build -t inference-perf-bench:latest -f paper/neurips/inference-perf.Dockerfile .
```

### 2. Run the llm-d inference simulator

```bash
kubectl apply -f llm-d-simulator.yaml
```

### 3. Deploy Jobs

Apply the job manifests:

```bash
kubectl apply -f vllm-job.yaml
kubectl apply -f inferencex-job.yaml
kubectl apply -f guidellm-job.yaml
kubectl apply -f aiperf-job.yaml
kubectl apply -f inference-perf-job.yaml
```

### 4. Verify Results

Check the logs of the jobs to see the benchmark results:

```bash
kubectl logs job/vllm-bench
kubectl logs job/inferencex-bench
kubectl logs job/guidellm-bench
kubectl logs job/aiperf-bench
kubectl logs job/inference-perf-bench
```

## Tool Versions

The Dockerfiles created for these experiments were built in April 2026. The following were the tool versions at that time.

| Tool | Version / Commit |
| --- | --- |
| **inference-perf** | `v0.4.0` |
| **MLPerf LoadGen** | `6.0.14` |
| **vLLM** | `0.20.0` |
| **InferenceX** | `504048f136a1e17b7896ef2fdc1d35874c5d37b4` |
| **AI Perf** | `0.7.0` |
| **Guide LLM** | `0.6.0` |
| **k6** | `grafana/k6@sha256:9481efe0c1e12ede6af3caae289026001106dd6fcd462c61fa1c35ea565cb958` |
| **Locust** | `locustio/locust@sha256:ea785ebc49c887007e0e6809cc9a839edc0d2199a4ddf1d249f23f11fda52787` |