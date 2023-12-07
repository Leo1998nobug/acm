# 模拟拍照并返回状态
from tools.camera import Camera
from tools.sound_collect import AudioProcess, recognize_audio_file
from tools.tcp_connect import TcpConnect


def take_photo():
    camera = Camera()
    camera.take_photo('test.jpg')
    return 1  # 假设拍照成功


# 模拟发送TCP指令并返回状态
def send_tcp_command():
    # 使用示例
    tcp_client = TcpConnect()  # 更改为你想要连接的主机和端口

    tcp_client.send_data("1")

    return 1  # 假设发送成功

def judge_photo():
    camera = Camera()
    camera.judge_color('test.jpg')


# 模拟收集声音并返回状态
def collect_sound():
    ad = AudioProcess()
    # ad.record_audio("recordaudio.wav", record_second=10)
    result = recognize_audio_file("recordaudio.wav")
    if "乘客" in result:
        return 1


if __name__ == '__main__':
    take_photo()
    judge_photo()
    send_tcp_command()
    take_photo()
    judge_photo()
    collect_sound()
