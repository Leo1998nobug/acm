# -*- coding:utf-8 -*-
# @Author:Leonardo(h541215)
# @Time:2023/11/5
# @file:camera.py
import time
import pandas as pd
import cv2


class Camera:

    def take_photo(self,photo):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        flag = cap.isOpened()
        index = 1
        # while (flag):
        ret, frame = cap.read()
        cv2.imshow("Capture_Paizhao", frame)
        k = cv2.waitKey(1) & 0xFF
        time.sleep(2)
        photo_name = photo
        cv2.imwrite(photo_name, frame)

    def judge_color(self,photo):

        index = ["color", "color_name", "hex", "R", "G", "B"]
        thefile = photo
        img = cv2.imread('test.jpg')
        led_1_x = 477
        led_1_y = 290
        b, g, r = img[led_1_y, led_1_x]
        b = int(b)
        g = int(g)
        r = int(r)
        csv_df = pd.read_csv('colors.csv', names=index, header=None)

        def get_color_name(r, g, b):
            min_diff = 10000
            color_name = ''
            for i in range(len(csv_df)):
                d = abs(r - int(csv_df.loc[i, "R"])) + abs(g - int(csv_df.loc[i, "G"])) + abs(
                    b - int(csv_df.loc[i, "B"]))
                if d <= min_diff:
                    min_diff = d
                    color_name = csv_df.loc[i, "color_name"]
            return color_name

        print(get_color_name(r, g, b))


