import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# 프롬프트 파일 경로 설정 (llm_handler.py 위치 기준)
BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompts" / "safety_guide_prompt.md"

def load_prompt_template(prompt_file_path: str = None) -> str:
    """.md 파일에서 프롬프트 템플릿을 읽어옵니다."""
    try:
        path_to_open = prompt_file_path if prompt_file_path else PROMPT_PATH
        with open(path_to_open, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        # 파일이 없을 경우를 대비한 폴백(Fallback) 기본 프롬프트
        return "당신은 운전자 보조 AI입니다. 탐지 요소: [{detection_summary}]. 3문장 이내 대화체로 요약하고 안전 주의/면책 문구를 포함해 답변해 주세요."

def get_gemini_api_key():
    """Streamlit Secrets 또는 로컬 .env에서 API Key를 가져옵니다."""
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")

def generate_safety_guide(
    detected_objects: dict,
    temperature: float = 0.7,
    prompt_file_path: str = None
) -> str:
    """YOLO 탐지 결과를 기반으로 Gemini LLM을 호출하여 운전자 안내 문구를 생성합니다."""
    
    api_key = get_gemini_api_key()
    if not api_key:
        return "⚠️ GEMINI_API_KEY가 설정되지 않았습니다. Secrets 또는 .env 파일을 확인해 주세요."

    if not detected_objects:
        return "현재 화면에서 탐지된 주요 도로 요소가 없습니다. 주변 도로 상황을 직접 확인하며 안전 운전하세요."

    # 1. YOLO 결과를 텍스트로 변환
    detection_summary = ", ".join([f"{obj} {count}개" for obj, count in detected_objects.items()])

    # 2. 프롬프트 데이터 바인딩
    prompt_template = load_prompt_template(prompt_file_path)
    prompt = prompt_template.format(detection_summary=detection_summary)

    # 3. Google Gemini API 호출
    try:
        client = genai.Client(api_key=api_key)
        
        # 파라미터 설정
        config = types.GenerateContentConfig(
            temperature=temperature
        )
        
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=config
        )
        return response.text.strip()
        
    except Exception as e:
        return f"현재 화면에서 [{detection_summary}]이(가) 탐지되었습니다. 주변 도로 상황은 항상 직접 확인해 주세요. (오류: {e})"