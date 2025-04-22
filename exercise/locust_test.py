import json
import random
from locust import HttpUser, task, between

def apply_body(prompt, system_prompt="You are an AI Assistant. Please answer the request kindly."):
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


cnt = 0
seed_list = list(range(10000))
class VLLMUser(HttpUser):
    wait_time = between(0.05, 0.1)  # 사용자 요청 간 간격 (초)

    @task
    def send_prompt(self):
        global cnt
        global seed_list
        file_path = "fairy_tale.txt"
        with open(file_path, 'r') as f:
            fairy_tale_bundle = "".join(f.readlines())
        fairy_tales = list(map(lambda x:x.strip(), fairy_tale_bundle.split("="*80)[:-1]))
        prompts = list(map(lambda x:"아래 영어로 된 동화를 100자 정도의 한글로 요약해주세요. \n\n[Input]"+x, fairy_tales))
        
        payload = {
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False,
            "seed": seed_list[cnt], 
        }
        payload.update(apply_body(prompts[cnt%len(prompts)]))
        cnt+=1
        response = self.client.post("/v1/chat/completions", json=payload)
        return_body = json.loads(response.content.decode())
