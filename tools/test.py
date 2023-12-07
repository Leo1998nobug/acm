import speech_recognition as sr


def recognize_audio_file(file_path):
    # 创建一个语音识别器对象
    recognizer = sr.Recognizer()

    try:
        # 从音频文件中读取语音数据
        with sr.AudioFile(file_path) as source:
            print("正在处理音频文件...")
            audio_data = recognizer.record(source)

            # 使用Google Web Speech API进行语音识别
            text = recognizer.recognize_google(audio_data, language="zh-CN")
            print("识别结果：", text)
            return text

    except sr.UnknownValueError:
        print("无法识别音频")
    except sr.RequestError as e:
        print(f"请求失败; {e}")


if __name__ == "__main__":
    # 传入音频文件的路径
    audio_file_path = "recordaudio.wav"
    recognize_audio_file(audio_file_path)
