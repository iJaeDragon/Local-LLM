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