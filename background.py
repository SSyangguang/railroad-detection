# 功能：通过帧数截取视频中感兴趣片段、对感兴趣片段进行求平均得到背景图

import cv2
import numpy as np


# 定义函数，对原始视频进行裁剪
# 输入：
# capture: 已经读取的视频
# begin: 起始帧数
# end: 终止帧数
# 输出：
# 在文件夹中存储截取后的视频，保存名称为videoCut.avi，帧数和分辨率保持原始视频规格
def videoCut(capture, begin, end):
    # 获取视频信息
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 从原始视频中截取关键部分段
    out = cv2.VideoWriter('videoCut.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    i = 0
    success, frame = capture.read()
    while success:
        success, frame = capture.read()
        if success:
            i += 1
            if (i > begin and i < end):  # 截取起始帧到终止帧之间的视频
                out.write(frame)
        else:
            print('end')
            break


# 定义函数，对截取视频进行求平均，得到背景
# 输入：
# capture: 已经读取的视频
# 输出：
# 在文件夹中存储求得平均后的背景图像，保存的名称为resultImage.jpg
def meanBackground(capture):
    # 获取视频长度
    frameNum = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    success, frame = capture.read()
    # 初始化平均背景图像，初始化图像为视频首帧图像
    meanFrame = frame
    # 在后续处理中为了防止数值溢出，先进行数据类型转化，转为float32型，在处理完成后在转化为unint8格式进行保存
    meanFrame = meanFrame.astype(np.float32)
    cv2.imshow('original image', meanFrame)
    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()
        if ret == True:
            tempframe = frame
            tempframe = tempframe.astype(np.float32)
            # 将所有帧的图像进行叠加
            cv2.accumulate(tempframe, meanFrame)

            cv2.imshow('original video', frame)
            cv2.imshow('temp frame', tempframe)
            cv2.imshow('mean video', meanFrame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

    # cv2.imshow('accumulate image', meanFrame)
    # cv2.waitKey(0)
    meanFrame = meanFrame / frameNum  # 对叠加后的图像进行求平均
    meanFrame = meanFrame.astype(np.uint8)  # 从float32转为uint8格式
    cv2.imshow('result image', meanFrame)
    cv2.waitKey(300)
    cv2.imwrite('resultImage.jpg', meanFrame)



# 读取原始视频文件
capture = cv2.VideoCapture('video1.mp4')
if not capture.isOpened:
    print('Unable to open: ' + capture.input)
    exit(0)
# 截取视频
videoCut(capture, 10, 500)
# 读取截取后的视频
capture = cv2.VideoCapture('videoCut.avi')
# 通过截取视频，使用平均值法求背景
meanBackground(capture)
# 处理完毕，释放内存
capture.release()

