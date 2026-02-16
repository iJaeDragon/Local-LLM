import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# 1️⃣ 베이스 모델 지정
model_name = "deepseek-ai/deepseek-coder-1.3b-base"

# 2️⃣ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3️⃣ 베이스 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # fp16 사용 → VRAM 절약, GTX 1660에서 필수
    device_map="auto"           # GPU 자동 할당
)

# 4️⃣ LoRA 설정
lora_config = LoraConfig(
    r=8,  
    # LoRA rank (보통 4~64 범위 사용)
    # 값 ↑ → 표현력 증가, 학습 성능 향상 가능 / VRAM 사용량 증가, 과적합 위험 ↑
    # 값 ↓ → 가볍고 안정적 / 복잡한 패턴 학습 한계
    # GTX 1660 (6GB) 기준: 8~16 권장 → 현재 8은 안전한 설정

    lora_alpha=16,  
    # LoRA scaling 계수 (보통 r의 1~2배, 8~128 범위 사용)
    # 값 ↑ → LoRA 영향력 증가 / 과적합 가능성 ↑
    # 값 ↓ → LoRA 반영 약함
    # r=8 기준 16은 일반적인 안정 설정

    target_modules=["q_proj", "v_proj"],  
    # 적용 레이어 선택
    # 범위: q_proj, k_proj, v_proj, o_proj 등 가능
    # 많이 적용할수록 성능 ↑ 가능 / VRAM 사용량 ↑
    # GTX 1660 기준 최소 적용(q,v) 추천

    lora_dropout=0.05,  
    # 0.0 ~ 0.3 사용
    # 값 ↑ → 과적합 방지 / 학습 속도 ↓ 가능
    # 값 ↓ → 빠른 학습 / 과적합 위험 ↑
    # 0.05는 소규모 데이터 기준 무난

    bias="none",  
    # "none", "all", "lora_only" 가능
    # none → VRAM 절약 (권장)
    # all → 성능 ↑ 가능 / 메모리 ↑

    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 5️⃣ JSONL 데이터셋 로드
dataset = load_dataset("json", data_files="learning.jsonl")

# 6️⃣ 프롬프트 포맷팅
def format_example(example):
    text = f"### 질문:\n{example['instruction']}\n\n### 답변:\n{example['response']}"
    return {"text": text}

dataset = dataset.map(format_example)

# 7️⃣ 토큰화
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256  
        # 보통 128~1024 사용
        # 값 ↑ → 긴 문맥 학습 가능 / VRAM 사용량 급증
        # 값 ↓ → 메모리 절약 / 긴 문장 잘림
        # GTX 1660 기준 256은 안정적인 선택
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# 8️⃣ 학습 설정
training_args = TrainingArguments(
    output_dir="./lora-output",

    per_device_train_batch_size=2,  
    # 보통 1~8 사용 (GPU VRAM에 따라 결정)
    # 값 ↑ → 학습 안정성 ↑ / VRAM 사용량 ↑
    # GTX 1660 기준 1~2 권장

    gradient_accumulation_steps=4,  
    # 1~32 사용
    # 값 ↑ → 실제 배치 효과 증가 / 학습 느려짐
    # 현재 설정: 2 x 4 = 실질 batch size 8 효과

    num_train_epochs=10,  
    # 1~20 사용
    # 값 ↑ → 데이터 적으면 과적합 가능성 ↑
    # 소규모 데이터면 3~10 권장

    logging_steps=10,

    save_strategy="epoch",

    fp16=True,  
    # GTX 1660에서는 필수 (VRAM 절약)

    report_to="none"
)

# 9️⃣ Trainer 생성 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()

# 🔟 LoRA 어댑터 저장
model.save_pretrained("./lora-output")
