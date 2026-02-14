# 로컬 LLM 세팅

## 로컬 LLM이란?

로컬 LLM(Local Large Language Model)은 클라우드 서버가 아닌 자신의 컴퓨터에서 직접 실행하는 대규모 언어 모델을 말한다.
ChatGPT나 Claude 같은 서비스는 인터넷을 통해 외부 서버에 요청을 보내지만, 로컬 LLM은 내 PC나 서버에서 독립적으로 작동한다.

## 클라우드를 사용하지 않으려는 이유

1. 내부 코드 유출 방지 가능
2. 오프라인 환경에서도 사용 가능
3. 내부 규칙이나 정보 파인 튜닝 가능

## 모델 선택 조건

- 상업적 이용이 가능할 것
- 한국어, 영어를 이용할 수 있을 것
- 다양한 프로그래밍 언어를 지원할 것
  - 최신 언어뿐만 아니라 레거시 언어도 지원할 것

## 딥시크 선택

## 세팅 방법

딥시크 세팅은 올라마라는 프로그램을 활용하기로 했다.

올라마는 로컬 환경에서 LLM을 쉽게 실행할 수 있도록 해주는 경량 플랫폼이다.
복잡한 설정 과정 없이 바로 사용할 수 있다.

### 설치 및 실행

1. 올라마 홈페이지(https://ollama.com/download)에서 자신의 운영체제에 맞는 버전을 다운로드한다.

2. 터미널에서 정상적으로 설치됐는지 확인한다.

```bash
> ollama -v
ollama version is 0.16.1
```

3. 올라마 공식 홈페이지의 Models 페이지에서 사용할 모델을 탐색한다. (https://ollama.com/search)

<img width="1905" height="943" alt="image" src="https://github.com/user-attachments/assets/9cc85e10-7788-44b0-ad49-b74ee3ae7294" />

4. 터미널에서 원하는 모델을 다운로드하고 실행한다.

```bash
> ollama run rcpsy2022/deepseek-coder-v2
```
