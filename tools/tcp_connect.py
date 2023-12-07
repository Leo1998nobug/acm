import socket
import time


class TcpConnect:
    def __init__(self):
        self.host = '192.168.1.5'  # 板子连上路由器之后分配的IP地址
        self.port = 23  # 板子的端口号,这个是固定烧写在板子里的,不用动
        self.client_socket = socket.socket()  # 创建连接板子的客户端socket对象
        self.client_socket.connect((self.host, self.port))  # 连接到板子上
        self.client_socket.send('!'.encode())  # 切换为'!'工作模式
        time.sleep(1)

    def intToHexStrTime(self, dtime):
        strhex = str(hex(dtime))[2:]
        if len(strhex) == 3:
            strhex = '[0' + strhex + ']'
        elif len(strhex) == 2:
            strhex = '[00' + strhex + ']'
        return strhex

    def send_data(self, data):

        self.client_socket.send(data.encode())  # J1点击头点一下,按下时间是50ms,在板子上出厂设置好的按下时间
        time.sleep(3)
