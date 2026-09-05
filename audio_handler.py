from io import BytesIO
from gtts import gTTS

def text_to_speech_bytes(text: str) -> BytesIO:
    """
    텍스트를 입력받아 gTTS로 음성 데이터(mp3)를 생성하고
    메모리 버퍼(BytesIO) 형태로 반환합니다. (디스크 저장 없이 처리)
    """
    if not text:
        return None

    try:
        # 한국어(lang='ko') 설정
        tts = gTTS(text=text, lang='ko')
        
        # 파일 저장 대신 메모리에 오디오 파일 담기
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        print(f"TTS 생성 중 오류 발생: {e}")
        return None