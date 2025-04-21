import json
import random
from locust import HttpUser, task, between, events

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

correct_count = 0
incorrect_count = 0

class VLLMUser(HttpUser):
    wait_time = between(0.1, 1)  # 사용자 요청 간 간격 (초)

    @task
    def send_prompt(self):
        global correct_count 
        global incorrect_count
        
        a = random.randint(0, 10000)
        b = random.randint(0, 10000)
        prompt =  f"What is the sum of {a} and {b}? Without the reason, please return the answer only. Do NOT return the answer as a perfect sentence, please return the answer as an integer only. "
        
        payload = {
            "max_tokens": 512,
            "temperature": 0.01,
            "stream": False,
            # "seed": 12345, 
        }
        payload.update(apply_body(prompt))
        response = self.client.post("/v1/chat/completions", json=payload)
        return_body = json.loads(response.content.decode())
        output = return_body["choices"][0]['message']['content']
        print(payload)
        print(output)
        try:
            print(a+b==int(output))
            correct_count += int(a+b==int(output))
            incorrect_count += (1-int(a+b==int(output)))
        except:
            print(False)
            incorrect_count += 1
        print("\n\n")

@events.quitting.add_listener
def _(environment, **kwargs):
    print("\n=== Test Summary ===")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {incorrect_count}")
    print("====================")
