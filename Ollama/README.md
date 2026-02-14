# Ollama

## Ollama란

모델 실행·관리 도구이며, 로컬 환경에서 LLM을 쉽게 실행할 수 있도록 해주는 경량 플랫폼이다.
기존에는 필요한 소프트웨어, 라이브러리, 모델 등을 직접 다운로드 받았어야 했었지만 이러한 과정을 생략하고 복잡한 설정 없이 바로 사용할 수 있다.
하지만 직접 파인튜닝이 불가능하다.

## 세팅 방법

### 설치 및 실행

1. 올라마 홈페이지(https://ollama.com/download)에서 자신의 운영체제에 맞는 버전을 다운로드한다.

2. 터미널에서 정상적으로 설치됐는지 확인한다.

```bash
> ollama -v
ollama version is 0.16.1
```

3. 올라마 공식 홈페이지의 Models 페이지에서 사용할 모델을 탐색한다. (https://ollama.com/search)

<img width="1905" height="943" alt="image" src="https://github.com/user-attachments/assets/9cc85e10-7788-44b0-ad49-b74ee3ae7294" />

4. 터미널에서 원하는 모델을 다운로드한다.

```bash
> ollama run rcpsy2022/deepseek-coder-v2
```

5. 다운로드 받은 모델을 실행한다.
   
```bash
> ollama run rcpsy2022/deepseek-coder-v2:latest
```

