import json
import random
from locust import HttpUser, task, between

def apply_body(prompt, system_prompt="You are an AI Assistant. Please answer the question kindly."):
    body = {
        "model": "models/Qwen2.5-1.5B-Instruct/",
        "messages": [
            {
                "content": system_prompt, 
                "role": "system"
            }, 
            {
                "content": prompt,
                "role": "user"
            }
        ],
    }
    return body


class VLLMUser(HttpUser):
    wait_time = between(0.1, 1)  # 사용자 요청 간 간격 (초)

    @task
    def send_prompt(self):
        prompts = [
            "Tell me a fun fact about space.",
            "What is the capital of Korea?",
            "Explain the theory of relativity simply in Korean.",
            "Translate 'Good morning' to Korean.",
            "Summarize the plot of Inception."
        ]
        
        payload = {
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": False,
            # "seed": 12345, 
        }
        payload.update(apply_body(random.choice(prompts)))
        response = self.client.post("/v1/chat/completions", json=payload)
        return_body = json.loads(response.content.decode())
        print(payload)
        print(return_body["choices"][0]['message']['content'], end="\n\n\n")
