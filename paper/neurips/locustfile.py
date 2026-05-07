from locust import FastHttpUser, task, constant_pacing
import os

class LLMUser(FastHttpUser):
    # Each user does 1 request per second
    wait_time = constant_pacing(1)
    
    @task
    def send_completion(self):
        model = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        payload = {
            "model": model,
            "prompt_token_ids": [1] * 320, # Dummy prompt of 320 tokens
            "max_tokens": 200,
            "ignore_eos": True,
        }
        headers = {"Content-Type": "application/json"}
        
        self.client.post("/v1/completions", json=payload, headers=headers)
