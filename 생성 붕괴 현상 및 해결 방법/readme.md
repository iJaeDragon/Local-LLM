# 생성 붕괴 현상 및 해결 방법

## 📚 목차
- [문제 상황: 붕괴 현상](#문제-상황-붕괴-현상)
- [붕괴 현상이란?](#붕괴-현상이란)
- [붕괴 증상](#붕괴-증상)
- [원인 분석](#원인-분석)
- [해결 방안](#해결-방안)

---

## 문제 상황: 붕괴 현상

인사 이후 짧은 대화를 이어가려 하였지만 이상한 답변을 받았다.

![생성 붕괴 예시](https://github.com/user-attachments/assets/71f88606-6865-4117-add3-84115b0be789)

---

## 붕괴 현상이란?

소형 모델에서 자주 발생하는 **생성 붕괴(Generation Collapse)** 또는 **문맥 이탈(Context Drift)** 현상이다.

대화가 길어지거나 복잡해질수록 모델이 문맥을 유지하지 못하고 의미 없는 텍스트를 반복하거나 엉뚱한 답변을 생성한다.

---

## 붕괴 증상

붕괴 현상이 발생하면 다음과 같은 증상이 나타난다:

1. **문맥 붕괴**: 이전 대화 내용을 무시하고 엉뚱한 답변
2. **의미 단절**: 문장이 중간에 끊기거나 의미가 연결되지 않음
3. **토큰 반복**: 같은 단어나 문장을 무한 반복
4. **문맥 이탈**: 질문과 전혀 관계없는 내용 생성

**예시:**
```
사용자: 리움에 대해 더 알려줘
모델: 리움 리움 리움 리움 리움 리움 리움...
```

---

## 원인 분석

현재 테스트는 **1.3B 초소형 모델**을 사용했다.

### 소형 모델의 한계

모델 크기가 작을수록 다음과 같은 한계가 있다:

- **파라미터 수 부족**: 복잡한 패턴 학습 불가
- **Attention head 수 부족**: 긴 문맥 추적 능력 약화
- **추론 능력 약화**: 논리적 연결성 유지 어려움

### 크기별 붕괴 빈도

| 모델 크기 | 붕괴 빈도 | 비고 |
|----------|---------|------|
| 1B ~ 2B  | 매우 잦음 | 3~5턴 이상 대화 시 높은 확률로 발생 |
| 7B       | 가끔    | 복잡한 질문이나 긴 대화에서 발생 |
| 13B      | 드뭄    | 일반적인 사용에서는 안정적 |
| 30B+     | 거의 없음 | 전문적 용도에서도 안정적 |

---

## 해결 방안

### 1. 더 큰 모델 사용 (권장)

**추천 모델:**
- **7B 모델**: 일반 사용자용 최소 권장 크기
  - `deepseek-coder-7b-base`
  - 대부분의 대화에서 안정적 동작
  
- **13B+ 모델**: 전문적 용도
  - 복잡한 논리 추론 필요 시
  - VRAM 16GB 이상 권장

### 2. 대화 길이 제한

최근 N개 턴만 컨텍스트로 사용:

```python
def chat_stream(message, history):
    # 최근 5턴(10개 메시지)만 유지
    recent_history = history[-10:] if len(history) > 10 else history
    
    conversation = ""
    for item in recent_history:
        # ... (기존 코드)
```

### 3. 생성 파라미터 조정

붕괴를 줄이는 파라미터 설정:

```python
generation_kwargs = dict(
    **inputs,
    max_new_tokens=150,           # 생성 길이 줄임 (200 → 150)
    temperature=0.5,              # 온도 높임 (0.3 → 0.5)
    top_p=0.9,                    # nucleus sampling 강화 (0.8 → 0.9)
    repetition_penalty=1.3,       # 반복 억제 강화 (1.1 → 1.3)
    no_repeat_ngram_size=3,       # 3-gram 반복 금지 (추가)
    do_sample=True,
    streamer=streamer
)
```

**파라미터 설명:**
- `no_repeat_ngram_size=3`: 연속된 3개 단어 조합 반복 방지
- `repetition_penalty=1.3`: 이미 나온 토큰에 패널티 부여

### 4. 프롬프트 엔지니어링

시스템 프롬프트 추가:

```python
def chat_stream(message, history):
    # 시스템 프롬프트로 행동 가이드 제공
    conversation = "### 시스템:\n간결하고 정확하게 답변하세요. 같은 내용을 반복하지 마세요.\n\n"
    
    for item in history:
        # ... (기존 코드)
```

### 5. 조기 중단 설정

반복 감지 시 생성 중단:

```python
# stopping_criteria 사용 (고급)
from transformers import StoppingCriteria, StoppingCriteriaList

class RepetitionStopper(StoppingCriteria):
    def __init__(self, tokenizer, max_repeats=3):
        self.tokenizer = tokenizer
        self.max_repeats = max_repeats
    
    def __call__(self, input_ids, scores, **kwargs):
        # 마지막 N개 토큰이 반복되면 중단
        # 구현 생략 (프로젝트 필요 시 추가)
        return False

# generation_kwargs에 추가
generation_kwargs = dict(
    **inputs,
    stopping_criteria=StoppingCriteriaList([RepetitionStopper(tokenizer)])
)
```

---

## 결론

**1.3B 모델의 한계:**
- 학습 및 테스트용으로는 적합
- 실제 서비스에는 부적합

**실용적 권장사항:**
- 최소 **7B 모델** 사용
- 대화 길이 제한 (5~10턴)
- 생성 파라미터 최적화

---
