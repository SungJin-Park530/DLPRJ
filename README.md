# Driving Environment Image Analysis

YOLOv8 기반 객체 탐지와 Gemini LLM 기반 안전 안내를 결합한 Streamlit 웹 애플리케이션입니다. 주행 환경 이미지를 분석해 교통 표지판을 탐지하고, 탐지 결과와 운전자의 음성 질문을 바탕으로 상황별 안내 문구를 제공합니다.

## 주요 기능

- JPG, JPEG, PNG, WEBP 형식의 주행 환경 이미지 업로드
- `samples/` 폴더의 예제 이미지를 무작위로 선택하여 빠르게 테스트
- YOLOv8 Nano / Medium 모델 선택 및 객체 탐지 결과 표시
- 탐지된 객체명, 신뢰도, 추론 시간 제공
- Gemini `gemini-3.6-flash`를 통한 탐지 결과 기반 안전 안내 생성
- 사이드바에서 LLM temperature와 안내 프롬프트 버전 조정
- 브라우저 마이크 녹음과 한국어 STT(Speech-to-Text) 지원
- 인식된 음성 질문과 탐지 결과를 함께 LLM에 전달해 추가 안내 생성
- 생성된 안내 문구의 한국어 TTS(Text-to-Speech) 자동 재생
- Precision, Recall, mAP50 기반 모델 성능 비교 시각화
- 모델 파일이 없을 때 Google Drive에서 자동 다운로드

## 서비스 화면

애플리케이션은 다음 두 탭으로 구성됩니다.

| 이미지 분석 | 학습 결과 및 비교 |
| --- | --- |
| 이미지를 업로드하거나 랜덤 예제 이미지를 선택한 뒤 모델을 골라 객체 탐지를 실행합니다. 탐지 결과를 바탕으로 AI 안전 안내를 받고, 마이크로 후속 질문을 할 수 있습니다. | Nano와 Medium 모델의 Precision, Recall, mAP50 및 파일 크기, 학습/추론 시간을 비교합니다. |

<!-- 아래 이미지 파일을 docs/images/에 추가한 후 주석을 해제하세요. -->

<!--
| 이미지 분석 화면 | 학습 결과 및 비교 화면 |
| --- | --- |
| ![이미지 분석 화면](docs/images/image-analysis-screen.png) | ![학습 결과 및 비교 화면](docs/images/model-comparison-screen.png) |
-->

## 시연 영상

서비스 실행부터 이미지 업로드, 모델 선택, 객체 탐지 결과 확인까지의 흐름을 담은 시연 영상을 추가할 자리입니다.

<!-- 동영상 파일을 docs/videos/에 추가하거나 YouTube 링크로 교체한 후 주석을 해제하세요. -->

<!--
[![주행 환경 이미지 분석 시연 영상](docs/images/demo-video-thumbnail.png)](https://www.youtube.com/watch?v=VIDEO_ID)
-->

## 기술 스택

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-111F68?logo=ultralytics&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Google%20Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?logo=googlegemini&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

| 구분 | 기술 |
| --- | --- |
| 언어 | Python 3.10+ |
| 객체 탐지 | Ultralytics YOLOv8 |
| 웹 대시보드 | Streamlit |
| 데이터 처리 | pandas, numpy |
| 시각화 | matplotlib |
| 이미지 처리 | Pillow, OpenCV |
| 생성형 AI | Google Gemini API (`gemini-3.6-flash`) |
| 음성 입력 | streamlit-mic-recorder, Google Speech Recognition (한국어 STT) |
| 음성 출력 | gTTS (한국어 TTS) |
| 모델 파일 관리 | gdown, Google Drive |
| 실험/학습 환경 | Jupyter Notebook |

## 시스템 동작 흐름

```mermaid
flowchart TD
    A[사용자] --> B[Streamlit 웹 UI]
    B --> C[이미지 업로드 또는 랜덤 예제 선택]
    C --> D{YOLOv8 모델 선택}
    D --> E[Nano 또는 Medium 가중치 로드]
    E --> F[객체 탐지 수행]
    F --> G[탐지 이미지, 객체명, 신뢰도, 추론 시간 표시]
    F --> H[탐지 객체별 개수 집계]
    H --> I[선택한 프롬프트와 temperature 적용]
    I --> J[Gemini 3.6 Flash]
    J --> K[안전 안내 문구 표시]
    K --> L[gTTS 한국어 음성 출력]

    A --> M[브라우저 마이크 녹음]
    M --> N[Google Speech Recognition 한국어 STT]
    N --> O[인식된 질문 표시]
    O --> P{이미지 분석 완료 여부}
    P -->|완료| Q[탐지 결과와 음성 질문을 Gemini에 전달]
    P -->|미완료| R[이미지 분석 먼저 안내]
    Q --> S[추가 안내 문구와 TTS 출력]
```

- 학습 및 분석: `박성진_딥러닝프로젝트.ipynb`에서 데이터 탐색과 모델 학습을 수행합니다.
- 서빙: `captcha_streamlit.py`가 YOLO 추론, Gemini 안내 생성, STT/TTS 처리 및 평가 지표 시각화를 제공합니다.
- LLM 처리: `llm_handler.py`가 탐지 결과와 선택한 프롬프트를 결합해 Gemini API를 호출합니다.
- 음성 처리: `audio_handler.py`가 Google Speech Recognition 기반 STT와 gTTS 기반 TTS를 처리합니다.
- 모델 관리: 첫 실행 시 `weights/` 폴더에 없는 가중치를 Google Drive에서 다운로드합니다.

## LLM 및 음성 상호작용

1. 이미지 분석을 실행하면 탐지된 표지판 종류와 개수가 Gemini에 전달되고 안전 안내 문구가 생성됩니다.
2. 사이드바에서 `Temperature`(0.0~1.0)와 프롬프트 버전을 선택할 수 있습니다.
   - 기본 안전 가이드: `prompts/safety_guide_prompt.md`
   - 간결 모드: `prompts/short_prompt.md`
3. 생성된 안내 문구는 화면에 표시되며 한국어 음성으로 재생됩니다.
4. 이미지 분석 후 마이크로 질문하면 한국어 STT 결과와 기존 탐지 결과를 함께 Gemini에 전달합니다.
5. 분석 전 음성 질문은 LLM에 전달하지 않고, 먼저 이미지 분석을 수행하도록 안내합니다.

## 프로젝트 구성

```text
.
├── captcha_streamlit.py            # Streamlit 웹 애플리케이션
├── llm_handler.py                   # Gemini API 호출 및 프롬프트 처리
├── audio_handler.py                 # 한국어 STT 및 TTS 처리
├── 박성진_딥러닝프로젝트.ipynb       # 데이터 분석 및 YOLO 모델 학습 노트북
├── requirements.txt                # 애플리케이션 실행 의존성
├── requirements(local).txt         # 로컬 학습/분석 환경 의존성
├── prompts/                        # Gemini 안내 프롬프트
│   ├── safety_guide_prompt.md
│   └── short_prompt.md
├── samples/                        # 서비스에서 사용하는 예제 이미지
└── weights/                        # 첫 실행 시 자동 다운로드되는 모델 가중치
    ├── v8n_best.pt
    └── v8m_best.pt
```

## 실행 방법

### 1. 사전 요구사항

- Python 3.10 이상
- pip

### 2. 저장소 클론

```powershell
git clone <repository-url>
cd DLPRJ
```

### 3. 가상환경 생성 및 활성화

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

PowerShell에서 실행 정책 오류가 발생하면 다음 명령을 실행한 뒤 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. 패키지 설치

```powershell
pip install -r requirements.txt
```

노트북에서 데이터 분석 또는 재학습까지 진행하려면 로컬 분석 환경 의존성을 설치합니다.

```powershell
pip install -r "requirements(local).txt"
```

### 5. Gemini API 키 설정

LLM 안전 안내 기능을 사용하려면 Google AI Studio에서 발급한 Gemini API 키가 필요합니다. 프로젝트 루트에 `.env` 파일을 만들고 다음 값을 설정합니다.

```dotenv
GEMINI_API_KEY=your_gemini_api_key
```

Streamlit Cloud 등 배포 환경에서는 `secrets.toml` 또는 Streamlit Secrets에 `GEMINI_API_KEY`를 등록할 수 있습니다.

Gemini API 키가 없더라도 YOLO 객체 탐지와 모델 성능 비교는 실행할 수 있으며, AI 안내 영역에는 키 설정 안내가 표시됩니다.

### 6. Roboflow 환경 변수 설정 (모델 학습 시)

데이터셋을 Roboflow에서 내려받아 노트북으로 학습하려면 프로젝트 루트에 `.env` 파일을 만들고, `.env.example`의 값을 본인 Roboflow 프로젝트 정보로 변경합니다.

```powershell
Copy-Item .env.example .env
```

`.env` 파일:

```dotenv
ROBOFLOW_API_KEY=your_roboflow_api_key
ROBOFLOW_WORKSPACE=your_workspace_name
ROBOFLOW_PROJECT=your_project_name
```

`ROBOFLOW_WORKSPACE`와 `ROBOFLOW_PROJECT`에는 Roboflow URL에 표시되는 워크스페이스 및 프로젝트 슬러그를 입력합니다. `.env`는 `.gitignore`에 포함되어 있어 API 키가 Git에 커밋되지 않습니다.

### 7. 애플리케이션 실행

```powershell
streamlit run captcha_streamlit.py
```

실행 후 브라우저에서 `http://localhost:8501`로 접속합니다. 최초 실행에서는 두 YOLO 모델 가중치가 `weights/` 폴더로 자동 다운로드되므로 인터넷 연결이 필요합니다. Gemini 안내, STT 및 TTS 기능도 외부 API를 사용하므로 인터넷 연결이 필요합니다.

## 모델 비교

| 모델 | Precision | Recall | mAP50 | 모델 파일 크기 | 학습 시간 | 예측 시간 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nano | 0.9252 | 0.8714 | 0.9430 | 6,088 KB | 약 5~6분 | 약 30ms |
| Medium | 0.9474 | 0.9122 | 0.9750 | 50,840 KB | 약 10~12분 | 약 70ms |

Medium 모델은 정확도 지표가 더 높고, Nano 모델은 파일 크기와 추론 시간이 더 작아 경량 환경에 적합합니다.

## 참고

- 본 프로젝트는 딥러닝 학습 및 포트폴리오 목적으로 작성되었습니다.
- `weights/`의 모델 가중치는 애플리케이션 실행 시 자동으로 내려받습니다.
- 데이터셋 출처와 사용 조건은 학습 노트북 및 데이터셋 원본의 라이선스를 확인해 주세요.