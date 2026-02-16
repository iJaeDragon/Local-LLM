# Transformers
# DeepSeek 모델 세팅

## 📚 목차
- [파이썬 & 라이브러리 세팅](#파이썬--라이브러리-세팅)
- [모델 탐색 및 다운로드](#모델-탐색-및-다운로드)
  - [1. Hugging Face 회원가입](#1-hugging-face-회원가입)
  - [2. 모델 선택](#2-모델-선택)
  - [3. 모델 다운로드](#3-모델-다운로드)
- [테스트 코드](#테스트-코드)
  - [주요 파라미터 설명](#주요-파라미터-설명)
- [Fine-tuning](#fine-tuning)
  - [Fine-tuning 방식 비교](#fine-tuning-방식-비교)
  - [LoRA Fine-tuning](#lora-fine-tuning)
    - [학습 데이터 준비](#학습-데이터-준비)
    - [LoRA 어댑터 생성](#lora-어댑터-생성)
    - [LoRA 모델 테스트](#lora-모델-테스트)

---

## 파이썬 & 라이브러리 세팅

필요한 라이브러리를 설치한다.

```bash
pip install --upgrade pip

# GPU 실행 - GPU 별로 버전 상이함 확인 후 진행
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 모델 로드
pip install transformers

# device 관리
pip install accelerate

# 토크나이저
pip install sentencepiece

# 모델 config
pip install protobuf

# 4bit/8bit 양자화
pip install bitsandbytes

# LoRA 학습
pip install peft

# 학습 데이터
pip install datasets

# 내부 계산
pip install scipy

# 텐서 연산
pip install einops
```

## 모델 탐색 및 다운로드

### 1. Hugging Face 회원가입

https://huggingface.co/ 에 접속하여 회원가입한다.

### 2. 모델 선택

사용할 모델을 탐색한다.
예시: https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base

#### 📊 선택 가이드

| 모델                | 파라미터 | 최소 VRAM | 적정 그래픽카드                                   | 특징                           | 추천도           |
| ----------------- | ---- | ------- | ------------------------------------------ | ---------------------------- | ------------- |
| **DeepSeek-1.3B** | 1.3B | 4~6GB   | GTX 1660 / RTX 2060                        | 응답 짧음, 복잡한 추론 약함, 문맥 유지 불안정  | 🔶 테스트용만 추천   |
| **DeepSeek-3B**   | 3B   | 8GB     | RTX 3060 12GB / RTX 4060                   | 기본 QA 가능, 긴 문맥은 약함, 코딩·추론 보통 | 🔷 가벼운 서비스 가능 |
| **DeepSeek-7B**   | 7B   | 12~16GB | RTX 3060 12GB(QLoRA) / RTX 3080 / RTX 3090 | 대화 안정적, 추론·요약 품질 양호, 실사용 가능  | ✅ 일반 서비스 추천   |
| **DeepSeek-13B**  | 13B  | 24GB    | RTX 3090 / RTX 4090                        | 긴 문맥 안정, 추론 품질 좋음, 생성 품질 높음  | ✅✅ 본격 서비스용    |
| **DeepSeek-33B**  | 33B  | 40GB+   | NVIDIA A40 / A100                          | 복잡한 추론 강함, 긴 컨텍스트 안정         | ⚠ 서버 전용       |
| **DeepSeek-67B**  | 67B  | 80GB+   | NVIDIA A100 80GB / H100                    | 고급 추론, 연구·기업급 활용             | ⚠ 기업/연구용      |

### 3. 모델 다운로드

1. 모델 페이지에서 "Use this model" 클릭
2. "Transformers" 클릭
3. 예제 코드를 확인하고 실행하여 로컬에 모델을 다운로드한다.

```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
```

다운로드한 모델 경로 : C:\Users\User\.cache\huggingface\hub

## 테스트 코드

다운로드한 모델을 테스트한다.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/deepseek-coder-1.3b-base"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=True  # 오프라인 모드
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    local_files_only=True  # 오프라인 모드
).to("cuda")

# 프롬프트 입력
prompt = "딥시크가 GPT보다 좋은 점"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 텍스트 생성
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

# 결과 출력
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### 주요 파라미터 설명

- `torch_dtype=torch.float16`: GPU 메모리 절약을 위한 16bit 연산 (GPU 사양에 따라 조정)
- `local_files_only=True`: 오프라인 모드 (다운로드된 모델만 사용)
- `max_new_tokens=200`: 생성할 최대 토큰 수
- `temperature=0.7`: 생성 다양성 조절 (낮을수록 보수적, 높을수록 창의적)

---

## Fine-tuning

### Fine-tuning 방식 비교

Fine-tuning 방식은 Full Model Fine-tuning과 LoRA(Low-Rank Adaptation) Fine-tuning 두 가지로 나뉘며, 각 특징은 다음과 같다.

| 방식 | 학습 범위 | VRAM/시간 | 장점 | 단점 |
|------|----------|-----------|------|------|
| Full Model Fine-tuning | 전체 weight | 매우 높음 | 모델 능력 그대로 사용 가능 | 학습 부담 ↑, 과적합 위험 높음 |
| LoRA Fine-tuning | 일부 어댑터 | 낮음 | VRAM 절약, 빠른 학습, 여러 LoRA 병행 가능 | 베이스 능력 밖 지식 학습 한계, 과적합 위험 존재 (단, Full보다 낮음) |

Full Fine-tuning은 사전학습된 베이스 모델의 모든 가중치(weight)를 업데이트하는 방식이다.
즉, 모델 내부의 수십억 개 파라미터를 전부 다시 학습한다.
이 방식은 모델의 표현 능력을 최대한 활용할 수 있다는 장점이 있다.
도메인이 크게 다르거나, 모델 구조 수준에서 적응이 필요한 경우에는 가장 강력한 방법이다.
하지만 VRAM 사용량이 매우 크며, 학습 시간이 오래 걸리고 저장 용량이 크다. 데이터가 적으면 과적합 위험도 존재

LoRA Fine-tuning은 베이스 모델의 가중치는 고정(freeze) 하고,
Attention 등의 일부 레이어에 작은 보조 행렬(저랭크 행렬)을 추가해 그 부분만 학습한다.
즉, 전체 모델을 바꾸는 것이 아니라 "출력 경로를 보정하는 작은 어댑터"를 학습하는 방식이다.

간단한 테스트를 위해 LoRA Fine-tuning 방식으로 진행했다.

---

## LoRA Fine-tuning

### 학습 데이터 준비

학습할 데이터를 JSONL 포맷으로 다음과 같은 형식으로 작성한다.

```json
{"instruction":"청록성의 주요 에너지원은 무엇인가?","response":"청록성의 주요 에너지원은 리움(Rium)이다."}
{"instruction":"리움은 어떤 성질을 가지고 있는가?","response":"리움은 감정에 반응해 밝기가 변하는 특성을 가진 에너지 물질이다."}
{"instruction":"청록성을 통치하는 조직은 무엇인가?","response":"청록성은 삼원 의회에 의해 통치된다."}
{"instruction":"삼원 의회는 몇 명으로 구성되는가?","response":"삼원 의회는 세 명의 대표로 구성된다."}
{"instruction":"청록성에서 새벽 3시 이후 금지되는 것은 무엇인가?","response":"새벽 3시 이후에는 인공 빛 사용이 금지된다."}
···
```

정상적으로 학습이 되었는지를 확인하기 위해서 AI가 알 수 없는 현실 세계에 존재하지 않는 내용을 입력하였다.(GPT로 내용 생성)

참고: 과적합(Overfitting) 문제를 방지하기 위해서는 많은 데이터를 생성해야 한다.

- 간단한 QA: 100~500건
- 도메인 특화 대화: 1,000~5,000건
- 복잡한 태스크: 10,000건 이상

---

## Fine-tuning

### Fine-tuning 방식 비교

Fine-tuning 방식은 Full Model Fine-tuning과 LoRA(Low-Rank Adaptation) Fine-tuning 두 가지로 나뉘며, 각 특징은 다음과 같다.

| 방식 | 학습 범위 | VRAM/시간 | 장점 | 단점 |
|------|----------|-----------|------|------|
| Full Model Fine-tuning | 전체 weight | 매우 높음 | 모델 능력 그대로 사용 가능 | 학습 부담 ↑, 데이터 적으면 과적합 ↑ |
| LoRA Fine-tuning | 일부 어댑터 | 낮음 | VRAM 절약, 빠른 학습, 여러 LoRA 병행 가능 | 베이스 능력 밖 지식 학습 한계 |

Full Fine-tuning은 사전학습된 베이스 모델의 모든 가중치(weight)를 업데이트하는 방식이다.
즉, 모델 내부의 수십억 개 파라미터를 전부 다시 학습한다.
이 방식은 모델의 표현 능력을 최대한 활용할 수 있다는 장점이 있다.
도메인이 크게 다르거나, 모델 구조 수준에서 적응이 필요한 경우에는 가장 강력한 방법이다.
하지만 VRAM 사용량이 매우 크며, 학습 시간이 오래 걸리고 저장 용량이 크다. 데이터가 적으면 과적합 위험도 존재

LoRA Fine-tuning은 베이스 모델의 가중치는 고정(freeze) 하고,
Attention 등의 일부 레이어에 작은 보조 행렬(저랭크 행렬)을 추가해 그 부분만 학습한다.
즉, 전체 모델을 바꾸는 것이 아니라 "출력 경로를 보정하는 작은 어댑터"를 학습하는 방식이다.

간단한 테스트를 위해 LoRA Fine-tuning 방식으로 진행했다.

---

## LoRA Fine-tuning

### 학습 데이터 준비

학습할 데이터를 JSONL 포맷으로 다음과 같은 형식으로 작성한다.

```json
{"instruction":"청록성의 주요 에너지원은 무엇인가?","response":"청록성의 주요 에너지원은 리움(Rium)이다."}
{"instruction":"리움은 어떤 성질을 가지고 있는가?","response":"리움은 감정에 반응해 밝기가 변하는 특성을 가진 에너지 물질이다."}
{"instruction":"청록성을 통치하는 조직은 무엇인가?","response":"청록성은 삼원 의회에 의해 통치된다."}
{"instruction":"삼원 의회는 몇 명으로 구성되는가?","response":"삼원 의회는 세 명의 대표로 구성된다."}
{"instruction":"청록성에서 새벽 3시 이후 금지되는 것은 무엇인가?","response":"새벽 3시 이후에는 인공 빛 사용이 금지된다."}
···
```

정상적으로 학습이 되었는지를 확인하기 위해서 AI가 알 수 없는 현실 세계에 존재하지 않는 내용을 입력하였다.

과적합(Overfitting) 문제를 방지하기 위해선 최소 2만건 이상에 데이터를 생성해야 한다.

### LoRA 어댑터 생성

다음 코드를 통해 LoRA Adapter를 생성한다.

```python
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
# LoRA는 "기존 모델 + 작은 어댑터" 방식이므로
# 반드시 사전에 학습된 베이스 모델이 필요함


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
    torch_dtype=torch.float16,  # FP16 사용 → VRAM 절약 (GTX 1660 필수)
    device_map="auto"           # GPU 자동 배치
)

# ==============================
# 4️⃣ LoRA 설정
# ==============================
lora_config = LoraConfig(
    r=8,
    # LoRA rank
    # 값이 클수록 표현력↑ / VRAM↑
    # 1660 기준: 8 ~ 16 권장

    lora_alpha=16,
    # LoRA scaling 계수
    # 보통 r의 2배가 안정적

    target_modules=["q_proj", "v_proj"],
    # Transformer Attention 내부에서
    # Query / Value projection에만 LoRA 적용
    # (성능 대비 메모리 효율 최적)

    lora_dropout=0.05,
    # 과적합 방지, 데이터 적으면 0.05~0.1 추천

    bias="none",
    # bias는 학습하지 않음 (메모리 절약)

    task_type="CAUSAL_LM"
    # 자동회귀 언어 모델 (GPT 계열)
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
    # 추론 시에도 동일한 포맷 사용해야 성능 유지
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
    # 입력 토큰을 그대로 정답(label)로 사용
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
    # LoRA 어댑터 저장 위치

    per_device_train_batch_size=2,
    # GPU 1개당 batch size
    # 1660 기준 2가 안전

    gradient_accumulation_steps=4,
    # batch 2 × 4 = 실질 batch 8 효과
    # VRAM 절약용 기법

    num_train_epochs=10,
    # 데이터셋을 몇 번 반복 학습할지
    # 소규모 데이터셋이면 10~20 권장

    logging_steps=10,
    # 10 step마다 loss 출력

    save_strategy="epoch",
    # epoch마다 체크포인트 저장

    fp16=True,
    # FP16 연산 활성화 (VRAM 절약)

    report_to="none"
    # wandb 등 외부 로깅 비활성화
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
# 베이스 모델은 저장 안 됨
# 이 폴더만 있으면 추론 가능
```

### LoRA 모델 테스트

LoRA 어댑터 생성이 완료되면 다음 코드로 테스트를 진행한다.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "deepseek-ai/deepseek-coder-1.3b-base"

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    base_model,
    "./lora-output"
)

model.eval()

# 테스트 질문
prompt = "### 질문:\n리움의 색은 무엇인가?\n\n### 답변:\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**출력 결과:**

```
### 질문:
리움의 색은 무엇인가?

### 답변:
리움의 색은 검은색이다.
```

---
