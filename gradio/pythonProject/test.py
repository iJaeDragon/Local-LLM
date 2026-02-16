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
        max_new_tokens=1000,
        temperature=0.7,
        do_sample=True
    )

# 결과 출력
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)