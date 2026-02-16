import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# ==============================
# 1️⃣ 베이스 모델 지정
# ==============================
model_name = "deepseek-ai/deepseek-coder-1.3b-base"
# → LoRA는 "기존 모델 + 작은 어댑터" 방식이므로
#   반드시 사전에 학습된 베이스 모델이 필요함


# ==============================
# 2️⃣ 토크나이저 로드
# ==============================
tokenizer = AutoTokenizer.from_pretrained(model_name)

# pad_token이 없는 모델이 많기 때문에
# eos_token을 padding으로 강제 지정 (학습 안정화용)
# pad_token을 eos_token으로 지정하지 않으면 학습 중 오류/비효율 발생
tokenizer.pad_token = tokenizer.eos_token


# ==============================
# 3️⃣ 베이스 모델 로드
# ==============================
# DeepSeek Coder 1.3B
# - GTX 1660 (6GB VRAM)에서 LoRA 학습 가능한 최대급
# - 코드 + 일반 텍스트 모두 안정적
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,     # FP16 사용 → VRAM 절약 (GTX 1660 필수)
    device_map="auto"        # GPU 자동 배치
)

# ==============================
# 4️⃣ LoRA 설정
# ==============================
lora_config = LoraConfig(
    r=8,
    # → LoRA rank
    #   값이 클수록 표현력↑ / VRAM↑
    #   1660 기준: 8 ~ 16 권장

    lora_alpha=16,
    # → LoRA scaling 계수
    #   보통 r의 2배가 안정적

    target_modules=["q_proj", "v_proj"],
    # → Transformer Attention 내부에서
    #   Query / Value projection에만 LoRA 적용
    #   (성능 대비 메모리 효율 최적)

    lora_dropout=0.05,
    # 과적합 방지, 데이터 적으면 0.05~0.1 추천

    bias="none",
    # → bias는 학습하지 않음 (메모리 절약)

    task_type="CAUSAL_LM"
    # → 자동회귀 언어모델 (GPT 계열)
)

# 베이스 모델에 LoRA 어댑터 장착
model = get_peft_model(model, lora_config)


# ==============================
# 5️⃣ JSONL 데이터셋 로드
# ==============================
# learning.jsonl :
# {"instruction": "...", "response": "..."} 형식
dataset = load_dataset(
    "json",
    data_files="learning.jsonl"
)


# ==============================
# 6️⃣ 프롬프트 포맷팅
# ==============================
def format_example(example):
    # 학습 시 입력 구조를 명확히 고정
    # → 추론 시에도 동일한 포맷 사용해야 성능 유지
    text = f"### 질문:\n{example['instruction']}\n\n### 답변:\n{example['response']}"
    return {"text": text}

dataset = dataset.map(format_example)


# ==============================
# 7️⃣ 토큰화 + labels 생성
# ==============================
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,          # 최대 길이 초과 시 자름
        padding="max_length",     # 고정 길이 패딩
        max_length=256            # GTX 1660 안정값, 고사양이면 512~1024 가능
    )

    # causal LM 학습의 핵심
    # → 입력 토큰을 그대로 정답(label)로 사용
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

dataset = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names
)


# ==============================
# 8️⃣ 학습 설정
# ==============================
training_args = TrainingArguments(
    output_dir="./lora-output",
    # → LoRA 어댑터 저장 위치

    per_device_train_batch_size=2,
    # → GPU 1개당 batch size
    #   1660 기준 2가 안전

    gradient_accumulation_steps=4,
    # → batch 2 × 4 = 실질 batch 8 효과
    #   VRAM 절약용 기법

    num_train_epochs=10,
    # → 데이터셋을 몇 번 반복 학습할지
    #   소규모 데이터셋이면 10~20 권장

    logging_steps=10,
    # → 10 step마다 loss 출력

    save_strategy="epoch",
    # → epoch마다 체크포인트 저장

    fp16=True,
    # → FP16 연산 활성화 (VRAM 절약)

    report_to="none"
    # → wandb 등 외부 로깅 비활성화
)


# ==============================
# 9️⃣ Trainer 생성
# ==============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)


# ==============================
# 🔟 학습 시작
# ==============================
trainer.train()


# ==============================
# 1️⃣1️⃣ LoRA 어댑터 저장
# ==============================
model.save_pretrained("./lora-output")
# → 베이스 모델은 저장 안 됨
# → 이 폴더만 있으면 추론 가능