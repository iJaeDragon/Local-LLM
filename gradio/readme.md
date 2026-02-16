# Gradio 채팅 인터페이스 구축

## 📚 목차
- [Gradio란?](#gradio란)
- [필요 라이브러리 설치](#필요-라이브러리-설치)
- [채팅 인터페이스 구현](#채팅-인터페이스-구현)
  - [전체 코드](#전체-코드)
  - [코드 상세 설명](#코드-상세-설명)
- [실행 및 결과](#실행-및-결과)

---

## Gradio란?

Gradio는 머신러닝 모델을 웹 인터페이스로 쉽게 만들어주는 파이썬 라이브러리다.

**주요 특징:**
- 몇 줄의 코드로 웹 UI 생성 가능
- 실시간 스트리밍 지원
- 로컬 및 공개 배포 모두 지원
- 채팅, 이미지, 음성 등 다양한 인터페이스 제공

---

## 필요 라이브러리 설치

```bash
pip install gradio
```

---

## 채팅 인터페이스 구현

### 전체 코드

```python
import torch
import gradio as gr
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

# 모델 설정
base_model_name = "deepseek-ai/deepseek-coder-1.3b-base"
lora_path = "./lora-output"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 베이스 모델 로드
# torch.float16: GPU 메모리를 절반으로 줄임
# device_map="auto": 사용 가능한 GPU에 자동 배치
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()  # 드롭아웃 등 학습 전용 기능 비활성화

def chat_stream(message, history):
    """
    실시간으로 텍스트를 생성하며 반환하는 채팅 함수
    
    Args:
        message (str): 사용자의 현재 입력 메시지
        history (list): 이전 대화 내역
            형식: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    
    Yields:
        str: 실시간으로 생성되는 응답 텍스트
    """
    conversation = ""
    
    # 대화 히스토리를 학습 시 사용한 프롬프트 형식(### 질문: / ### 답변:)으로 변환
    for item in history:
        if item["role"] == "user":
            conversation += f"### 질문:\n{item['content']}\n\n"
        elif item["role"] == "assistant":
            conversation += f"### 답변:\n{item['content']}\n\n"
    
    conversation += f"### 질문:\n{message}\n\n### 답변:\n"
    
    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    
    # 실시간 스트리밍을 위한 TextIteratorStreamer 생성
    # skip_prompt=True: 프롬프트는 출력하지 않음
    # skip_special_tokens=True: 특수 토큰 제외
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # 생성 파라미터 설정
    # max_new_tokens: 생성할 최대 토큰 수 (200~500 권장)
    # temperature: 생성 다양성 (0.3~0.7 권장, 낮을수록 일관적)
    # top_p: nucleus sampling 확률 (0.8~0.95 권장)
    # repetition_penalty: 반복 방지 강도 (1.1~1.3 권장)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        top_p=0.8,
        repetition_penalty=1.1,
        do_sample=True,
        streamer=streamer
    )
    
    # 별도 스레드에서 생성 시작 (메인 스레드 블로킹 방지)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 실시간으로 생성된 텍스트를 yield (타이핑 효과)
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

demo = gr.ChatInterface(
    fn=chat_stream,
    title="DeepSeek Chat"
)

if __name__ == "__main__":
    demo.launch()
```

---

### 코드 상세 설명

위 코드의 주요 동작 원리를 정리하면 다음과 같다.

#### 1. 모델 로드

```python
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float16,     # FP16 사용 → VRAM 절약 (GTX 1660 필수)
    device_map="auto"        # GPU 자동 배치
)
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()  # 학습 모드 비활성화
```

- `torch.float16`: GPU 메모리를 절반으로 줄임
- `device_map="auto"`: 사용 가능한 GPU에 자동 배치
- `model.eval()`: 드롭아웃 등 학습 전용 기능 비활성화

#### 2. 대화 히스토리 처리

```python
for item in history:
    if item["role"] == "user":
        conversation += f"### 질문:\n{item['content']}\n\n"
    elif item["role"] == "assistant":
        conversation += f"### 답변:\n{item['content']}\n\n"
```

Gradio의 `ChatInterface`는 대화를 다음 형식으로 전달한다:
```python
[
    {"role": "user", "content": "안녕?"},
    {"role": "assistant", "content": "안녕하세요!"},
    ...
]
```

이를 학습 시 사용한 프롬프트 형식(`### 질문:` / `### 답변:`)으로 변환한다.

#### 3. 실시간 스트리밍

```python
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for new_text in streamer:
    partial_text += new_text
    yield partial_text
```

**스트리밍 동작 원리:**
1. `TextIteratorStreamer`: 생성된 토큰을 실시간으로 반환하는 이터레이터
2. `Thread`: 별도 스레드에서 생성 수행 (메인 스레드 블로킹 방지)
3. `yield`: 생성된 텍스트를 즉시 화면에 표시

#### 4. 생성 파라미터

| 파라미터 | 설명 | 권장값 |
|---------|------|--------|
| `max_new_tokens` | 생성할 최대 토큰 수 | 200~500 |
| `temperature` | 생성 다양성 (0~2) | 0.3~0.7 |
| `top_p` | nucleus sampling 확률 | 0.8~0.95 |
| `repetition_penalty` | 반복 방지 강도 | 1.1~1.3 |

**temperature 값에 따른 변화:**
- `0.1~0.3`: 일관적이고 안정적 (QA, 사실 기반 응답)
- `0.7~0.9`: 창의적이고 다양함 (스토리텔링, 브레인스토밍)
- `1.0 이상`: 매우 자유로움 (실험적 용도)

---

## 실행 및 결과

### 접속 URL

터미널에 다음과 같이 표시된다:

```
Running on local URL:  http://127.0.0.1:7860
```

브라우저에서 해당 주소로 접속하면 채팅 인터페이스를 사용할 수 있다.

### 결과 화면

![DeepSeek Chat Interface](https://github.com/user-attachments/assets/5a6394ff-b4bb-4650-a7b8-6828b403d599)

**주요 기능:**
- 실시간 텍스트 스트리밍 (타이핑 효과)
- 대화 히스토리 유지
- 메시지 복사/삭제 기능
- 다크/라이트 모드 전환

### 공개 배포 (선택)

외부에서 접속 가능한 임시 URL을 생성하려면:

```python
demo.launch(share=True)
```

72시간 동안 유효한 공개 링크가 생성된다.

---
