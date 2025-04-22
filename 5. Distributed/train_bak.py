import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from accelerate import Accelerator

def main():
    # 모델과 토크나이저 로드
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )

    # PEFT 설정
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 모델 준비
    model = get_peft_model(model, peft_config)

    # 데이터셋 로드 및 전처리
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    dataset = load_dataset("Junnos/luckyvicky")  # 실제 데이터셋으로 교체 필요
    # if int(os.envionr["RANK"])==0:
    print(f"Dataset load 직후: {dataset['train'][0]}")
    dataset = dataset.map(lambda x: {'text': x['input'] + x['output']})
    print(f"Text column 추가: {dataset['train'][0]}")
    tokenized_dataset = dataset.map(preprocess_function, 
                                    batched=True,
                                   )
    print(f"Tokenizer 통과: {tokenized_dataset['train'][0]}")

    # Accelerator 설정
    accelerator = Accelerator()

    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./train_results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_steps=100,
        logging_steps=10,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
    )

    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    # 학습 실행
    trainer.train()

    # 모델 저장
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")

if __name__ == "__main__":
    main() 