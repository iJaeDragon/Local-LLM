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
