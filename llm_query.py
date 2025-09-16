import os
import tempfile
import pyaudio
import soundfile as sf
from google.cloud import speech
from google.cloud import texttospeech as tts
from playsound import playsound
import google.generativeai as genai

# Google Cloud 클라이언트 초기화
speech_client = speech.SpeechClient()
tts_client = tts.TextToSpeechClient()

# Google Gemini API 키 설정 (실제 키로 교체 필요)
GOOGLE_API_KEY = 'YOUR_API_KEY'  # 여기에 실제 API 키를 입력하세요.
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')
generation_config = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

# 음성 녹음 함수
def record_audio(filename, duration=5, rate=16000):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=rate, input=True,
                        frames_per_buffer=CHUNK)
    
    print("️ 녹음 중...")
    frames = []
    for _ in range(0, int(rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("✅ 녹음 완료.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with sf.SoundFile(filename, 'w', samplerate=rate, channels=CHANNELS, subtype='PCM_16') as f:
        f.write(b''.join(frames))

# Google Speech-to-Text API를 사용하여 음성을 텍스트로 변환
def transcribe_audio_google(filename):
    print("Google STT로 변환 중...")
    with open(filename, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR"
    )

    response = speech_client.recognize(config=config, audio=audio)

    if not response.results:
        print("음성을 인식할 수 없습니다.")
        return ""

    transcript = response.results[0].alternatives[0].transcript
    print(f"인식된 텍스트: {transcript}")
    return transcript

# Google Text-to-Speech API를 사용하여 텍스트를 음성으로 변환
def speak_text_google(text, output_filename):
    print("Google TTS로 음성 합성 중...")
    synthesis_input = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Wavenet-A"
    )
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.MP3
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
        print(f'음성 파일 저장 완료: {output_filename}')

# LLM 쿼리 함수
def query_llm(text, model_pred_txt):
    full_prompt = f"""당신은 의사 보조 인공지능입니다. 진단 모델은 해당 이미지에 대해 {model_pred_txt} 라고 예측했습니다.
                의사는 다음과 같은 피드백을 제공했습니다: {text}

                이에 따라 진단을 어떻게 수정할지, 아래 형식에 맞추어 간결하게 한 문장으로 한국어로 대답하세요. (단, 질병 이름은 영어로 유지하십시오.)

                - 의사에게 피드백에 대해 간단히 인사합니다.
                - 현재 진단 클래스(Class[숫자])가 무엇이었는지 명확히 밝히고,
                - 의사 피드백에 근거하여 어떤 클래스(Class[숫자])로 바꾸고 싶은지 설명하세요.

                다음은 클래스 번호, 약어, 실제 질병 이름의 매칭 정보입니다:
                (1, AKIEC, Actinic keratosis), 
                (2, BCC, Basal cell carcinoma), 
                (3, BKL, Benign keratosis), 
                (4, DF, Dermatofibroma), 
                (5, MEL, Melanoma), 
                (6, NV, Melanocytic nevi), 
                (7, VASC, Vascular skin lesions)

                만약 피드백과 정확히 일치하는 질병이 없다면, 주어진 클래스 정보를 바탕으로 유추하여 적절한 클래스를 선택하세요.
                """
    response = model.generate_content(full_prompt, generation_config=generation_config)
    return response.text.strip()

# 전체 파이프라인 실행
def run_pipeline():
    model_pred_txt = '환자의 피부 질환에 대한 진단 결과는 다음과 같습니다. 진단한 질병명은 Melanoma이며, 진단에 대한 신뢰도는 64%입니다.'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
        speak_text_google(model_pred_txt, tmp_audio_file.name)
        playsound(tmp_audio_file.name)
        os.remove(tmp_audio_file.name)

    # --- 사용자 입력 대기 부분 --- #
    input("사용자 응답을 시작하려면 Enter 키를 누르세요...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
        record_audio(tmp_audio_file.name, duration=5)
        user_text = transcribe_audio_google(tmp_audio_file.name)
    
    os.remove(tmp_audio_file.name)

    if user_text:
        response_text = query_llm(user_text, model_pred_txt)
        print(f"LLM 응답: {response_text}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
            speak_text_google(response_text, tmp_audio_file.name)
            playsound(tmp_audio_file.name)
            os.remove(tmp_audio_file.name)
    else:
        print("음성 입력이 없어 다음 단계를 진행하지 않습니다.")

if __name__ == "__main__":
    run_pipeline()