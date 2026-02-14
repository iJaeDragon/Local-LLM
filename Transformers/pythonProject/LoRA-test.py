import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "deepseek-ai/deepseek-coder-1.3b-base"

# 🔹 토크나이저
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🔹 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 🔹 LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    base_model,
    "./LoRA/lora-output"
)

model.eval()

# 🔹 테스트 질문
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