import http from 'k6/http';
import { check } from 'k6';

export const options = {
  scenarios: {
    constant_request_rate: {
      executor: 'constant-arrival-rate',
      rate: parseInt(__ENV.QPS) || 1000, // Target QPS
      timeUnit: '1s',
      duration: '300s',
      preAllocatedVUs: 100, // Adjust based on expected concurrency
      maxVUs: 1000,
    },
  },
};

export default function () {
  const url = __ENV.URL || 'http://vllm-llama3-8b-instruct-svc:8000';
  const model = __ENV.MODEL || 'meta-llama/Llama-3.1-8B-Instruct';
  
  const payload = JSON.stringify({
    model: model,
    prompt_token_ids: Array(320).fill(1), // Dummy prompt of 320 tokens
    max_tokens: 200,
    temperature: 0,
    ignore_eos: true,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post(`${url}/v1/completions`, payload, params);
  
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}
