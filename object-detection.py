import numpy as np
import cv2

# 定义前景背景相减的函数
def subtract(frame, background):
    # 前景和帧图像相减
    mask = frame - background
    # 对相减后的图像进行二值化
    ret, binaryMask = cv2.threshold(mask, 210, 255, cv2.THRESH_BINARY)
    # 对相减后的图像进行自适应阈值化
    binaryMaskAdative = cv2.adaptiveThreshold(mask, 255, \
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)

    return binaryMask

# 读取视频
capture = cv2.VideoCapture('videoCut.avi')
# 读取背景图像转为灰度图
background = cv2.imread('resultImage.jpg')
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background = cv2.GaussianBlur(background, (3, 3), 15, 1)
if not capture.isOpened:
    print('Unable to open: ' + capture.input)
    exit(0)

success, frame = capture.read()


while success:
    success, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 帧图像与背景图像相减
    output = subtract(frame, background)

    cv2.imshow('background', background)
    cv2.imshow('frame', frame)
    cv2.imshow('mask image', output)
    # cv2.imshow('adative mask image', binaryMaskAdative)
    k = cv2.waitKey(300) & 0xff
    if k == 27:
        break


capture.release()
cv2.destroyAllWindows()