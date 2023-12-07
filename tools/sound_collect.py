import time

import pyaudio
import wave
from tqdm import tqdm
import numpy as np
import speech_recognition as sr


class AudioProcess():
    def __init__(self):
        pass

    def record_audio(self, wave_out_path, record_second=2):
        '''
        接收从Dom2过来的音源信号，进行录音
        :param wave_out_path: 录音文件存放的地址
        :param record_second: 录音的时间
        :return:
        '''
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        '''
        声卡声道道数量1.单声道, 2.双声道
        '''
        RATE = 44100
        '''
        声卡DefaultSampleRate要和硬件匹配，pyaudio.get_device_info_by_index(i)查询DefaultSampleRate，CHANNELS 和input Device index信息
        '''
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=None,
                        frames_per_buffer=CHUNK)
        '''
        input Device index是驱动索引号要对应上mic输入的驱动索引,None是可以指向系统设置里面的默认input选项
        '''
        wf = wave.open(wave_out_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        print("* recording")
        for i in tqdm(range(0, int(RATE / CHUNK * int(record_second)))):
            data = stream.read(CHUNK)
            # print(data)
            wf.writeframes(data)
        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
        return wave_out_path

    def checkFFTSoundExist(self, wave_out_path):
        '''
         检查音源的FFT返回最大幅值，并根据读取的数据判断是否跟源声音的幅度一致，从而判断是否有声音
        :param wave_out_path: 录音文件存放的地方
        :return: 福值
        '''
        # 打开wav文件 ，open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据。
        f = wave.open(wave_out_path, "rb")
        # 读取格式信息
        # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采
        # 样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # 读取波形数据
        # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
        str_data = f.readframes(nframes)
        f.close()
        # 将波形数据转换成数组
        # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
        wave_data = np.frombuffer(str_data, dtype=np.short)
        # 通过取样点数和取样频率计算出每个取样的时间。
        time = np.arange(0, nframes) / framerate
        # 由于plot函数里的两个参数维度不同，于是将time的长度除以2。
        len_time = len(time) // 2
        time = time[0:len_time]
        num = 0  # 定义1000幅度出现的次数
        fft_size = 512  # FFT处理的取样长度
        deltaf = framerate / fft_size  # 频率分辨率
        # print(deltaf)
        data1 = self.enframe(wave_data, fft_size, fft_size)  # 安装fft的取样长度分帧处理所有的音频数据
        for index, item in enumerate(data1):
            xs = item  # 从每一帧波形数据中取样fft_size个点进行运算
            xf = np.fft.rfft(xs) / fft_size  # 利用np.fft.rfft()进行FFT计算，rfft()是为了更方便对实数信号进行变换，由公式可知/fft_size为了正确显示波形能量
            # rfft函数的返回值是N/2+1个复数，分别表示从0(Hz)到sampling_rate/2(Hz)的分。
            # 于是可以通过下面的np.linspace计算出返回值中每个下标对应的真正的频率：
            freqs = np.linspace(0, framerate / 2, int(fft_size / 2 + 1))
            # np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
            # 在指定的间隔内返回均匀间隔的数字
            # xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
            xfp = np.clip(np.abs(xf), 1e-20, 1e100)
            # 最后我们计算每个频率分量的幅值，并通过 20*np.log10()将其转换为以db单位的值。为了防止0幅值的成分造成log10无法计算，我们调用np.clip对xf的幅值进行上下限处理
            # print(params)
            # print("time length = ", len(time))
            # print("wave_data[0] length = ", len(wave_data))
            # 取1~(xfp的长度/2)-1的值来计算最大复制的索引号
            nxfp = int(xfp.size / 2) - 1
            fxfp = xfp[1:nxfp]
            maxindex = np.argmax(fxfp)  # 找到最大的幅值的索引号
            lmaxindex = np.argmax(fxfp) - 1  # 找到最大的幅值的索引号-1
            rmaxindex = np.argmax(fxfp) + 1  # 找到最大的幅值的索引号+1

            # print(maxindex)
            # f1,f2,f3是幅度
            f1 = maxindex * deltaf
            f2 = lmaxindex * deltaf
            f3 = rmaxindex * deltaf
            print(f1, f2, f3)
            print(type(f1))
            print('fudu1:=' + str(abs(xf[maxindex])))
            # print('fudu2:=' + str(abs(xf[lmaxindex])))
            # print('fudu3:=' + str(abs(xf[rmaxindex])))
            # if float(f1)==1033.59375 or float(f2)==947.4609375 or float(f3)==1119.7265625:
            #     #幅值
            #     if abs(xf[maxindex])>10:
            #         num=num+1
            # if abs(xf[maxindex]) > 10:
            if float(f1) == 0.0 or float(f2) == -93.75 or float(f3) == 93.75:
                # if float(f1)==1406.25 or float(f2)==947.4609375 or float(f3)==1119.7265625:
                # if float(f1)==1033.59375 or float(f2)==947.4609375 or float(f3)==1119.7265625:
                num = num + 1

        # # # 绘图显示结果
        #  pl.xlabel("time(s)")
        #  pl.title(u"WaveForm And Freq")
        #  pl.subplot(311)
        #  pl.plot(time[:fft_size], xs)
        #  pl.xlabel(u"Time(S)")
        #  pl.subplot(312)
        #  pl.plot(freqs, xfp)
        #  pl.xlabel(u"Freq(Hz)")
        #  pl.subplots_adjust(hspace=0.4)
        #  pl.show()
        #  # return f1,f2,f3
        #  print(num)
        print('num=' + str(num))
        if num > 10:  # 如果取样检测的1000幅度出现的次数大于10次就认为跟源声音一致
            # log.info('Sound can be heared!')
            print('Sound can be heared!')
            return True
        else:
            # log.error('There is no sound!Please check devices!')
            print('There is no sound!Please check devices!')
            return False, 'There is no sound!Please check devices!'

    def monitor_audio(self):
        """
            PC声卡 MIC INput 声音监测
        """
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        SoundExit = 1000
        p = pyaudio.PyAudio()
        num = 0
        result = ''
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=None,
                        frames_per_buffer=CHUNK)
        print("Start monitoring")
        for i in range(0, 4):
            data = stream.read(1024)
            audio_data = np.fromstring(data, dtype=np.short)
            temp = np.max(audio_data)
            print(temp)
            result += "".join(str(temp) + "\n")
            time.sleep(1)
        result = result[:-1].strip()
        stream.stop_stream()
        stream.close()
        p.terminate()
        return result
        #     if i >= 4:
        #         if i == 4:
        #           print("Start checking the sound data:")
        #         elif temp >= SoundExit:
        #           num = num+1
        #           print('Current sound data：', temp)
        #           print('num '+str(num))
        #           if num >= 5:
        #             print("Monitor 10 second,Sound can be monitored!")
        #             return True
        # print("Monitor 10 second,There is no sound!Please make sure there is sound from speakerline or check audio devices and connection!")
        # return False

    def checkFFTSoundNotExist(self, wave_out_path):
        '''
        检查音源的FFT返回最大幅值，并根据读取的数据判断是否跟源声音的幅度一致，从而判断是否有声音
        1kHZ：1033.59375 ---1.2kHZ:1205.859375--1.5Khz:1464.2578125---1.8kHZ:1808.7890625--2kHz:1981.0546875--2.2kHZ:1981.0546875--2.5kHZ:2497.8515625--2.8kHZ:2842.3828125
        :param wave_out_path: 录音文件存放的地方
         :param min: 要检测的频率最小范围比如1khz，最小范围是1000
         :param max: 要检测的频率最大范围比如1khz，最大范围是1050
        :return: 频率
        '''
        # 打开wav文件 ，open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据。
        f = wave.open(wave_out_path, "rb")
        # 读取格式信息
        # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采
        # 样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # 读取波形数据
        # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
        str_data = f.readframes(nframes)
        f.close()
        # 将波形数据转换成数组
        # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
        wave_data = np.frombuffer(str_data, dtype=np.short)
        # 通过取样点数和取样频率计算出每个取样的时间。
        time = np.arange(0, nframes) / framerate
        # 由于plot函数里的两个参数维度不同，于是将time的长度除以2。
        len_time = len(time) // 2
        time = time[0:len_time]
        num = 0  # 定义1000幅度出现的次数
        fft_size = 512  # FFT处理的取样长度
        deltaf = framerate / fft_size  # 频率分辨率
        # print(deltaf)
        data1 = self.enframe(wave_data, fft_size, fft_size)  # 安装fft的取样长度分帧处理所有的音频数据
        for index, item in enumerate(data1):
            xs = item  # 从每一帧波形数据中取样fft_size个点进行运算
            xf = np.fft.rfft(xs) / fft_size  # 利用np.fft.rfft()进行FFT计算，rfft()是为了更方便对实数信号进行变换，由公式可知/fft_size为了正确显示波形能量
            # rfft函数的返回值是N/2+1个复数，分别表示从0(Hz)到sampling_rate/2(Hz)的分。
            # 于是可以通过下面的np.linspace计算出返回值中每个下标对应的真正的频率：
            freqs = np.linspace(0, framerate / 2, int(fft_size / 2 + 1))
            # np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
            # 在指定的间隔内返回均匀间隔的数字
            # xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
            xfp = np.clip(np.abs(xf), 1e-20, 1e100)
            # 最后我们计算每个频率分量的幅值，并通过 20*np.log10()将其转换为以db单位的值。为了防止0幅值的成分造成log10无法计算，我们调用np.clip对xf的幅值进行上下限处理
            # print(params)
            # print("time length = ", len(time))
            # print("wave_data[0] length = ", len(wave_data))

            # 取1~(xfp的长度/2)-1的值来计算最大复制的索引号
            nxfp = int(xfp.size / 2) - 1
            fxfp = xfp[1:nxfp]
            maxindex = np.argmax(fxfp)  # 找到最大的幅值的索引号
            lmaxindex = np.argmax(fxfp) - 1  # 找到最大的幅值的索引号-1
            rmaxindex = np.argmax(fxfp) + 1  # 找到最大的幅值的索引号+1
            # print(maxindex)
            # f1,f2,f3是幅度
            f1 = maxindex * deltaf
            f2 = lmaxindex * deltaf
            f3 = rmaxindex * deltaf
            print(f1, f2, f3)
            if float(f1) == 0.0 or float(f2) == -93.75 or float(f3) == 93.75:
                # if float(f1) == 1033.59375 or float(f2) == 947.4609375 or float(f3) == 1119.7265625 :
                num = num + 1
        if num > 1:  # 如果取样检测的1000幅度出现的次数大于10次就认为跟源声音一致
            # log.info('There is no sound!Please check devices!')
            return True
        else:
            # log.error('Sound can be heared!')
            return False, 'Sound can be heared!'

    def enframe(self, signal, nw, inc):
        '''将音频信号转化为帧。
        参数含义：
        signal:原始音频型号
        nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
        inc:相邻帧的间隔（同上定义）
        '''
        signal_length = len(signal)  # 信号总长度
        if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
            nf = 1
        else:  # 否则，计算帧的总长度
            nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
        pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
        zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
        pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
        indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                               (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
        indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
        frames = pad_signal[indices]  # 得到帧信号
        return frames

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



if __name__ == '__main__':
    ad = AudioProcess()
    ad.record_audio("recordaudio.wav", record_second=10)
    # print(ad.checkFFTSoundExist("22k.wav"))
    print(ad.checkFFTSoundExist(r"recordaudio.wav"))
