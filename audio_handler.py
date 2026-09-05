from io import BytesIO
from gtts import gTTS
import speech_recognition as sr

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

def speech_to_text_from_bytes(audio_bytes: bytes) -> str:
    """
    브라우저 마이크에서 전달받은 오디오 바이트(WAV) 데이터를
    Google Speech Recognition(무료)을 통해 한국어 텍스트로 변환합니다.
    """
    if not audio_bytes:
        return ""

    recognizer = sr.Recognizer()
    
    try:
        # 오디오 바이트 데이터를 SpeechRecognition용 AudioFile로 읽기
        audio_file = BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            
        # Google Web Speech API 호출 (무료, 한국어 'ko-KR')
        text = recognizer.recognize_google(audio_data, language='ko-KR')
        return text
    except sr.UnknownValueError:
        return "⚠️ 음성을 인식하지 못했습니다. 다시 명확하게 말씀해 주세요."
    except sr.RequestError as e:
        return f"⚠️ 음성 인식 서비스 연결 실패: {e}"
    except Exception as e:
        return f"⚠️ 처리 중 오류 발생: {e}"