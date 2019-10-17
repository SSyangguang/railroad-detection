import cv2
import numpy as np
import background



filename = "Rail-stable.mp4"
# 读取原始视频文件
capture = cv2.VideoCapture(filename)
if not capture.isOpened:
    print('Unable to open: ' + capture.input)
    exit(0)

# 输出视频的帧数和视频的尺寸
fps = capture.get(cv2.CAP_PROP_FPS)
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("输入视频的帧数为：", fps)
print("输入视频的尺寸为：", size)

# 截取原始视频并对视频长度进行裁减
background.videoCut(capture, 1, 210)
# 读取截取后的视频
captureCut = cv2.VideoCapture('videoCut.avi')
# 通过截取视频，使用平均值法求背景
background.meanBackground(captureCut)
# 处理完毕，释放内存
capture.release()

# 初步的视频裁减完毕

# 开始通过保存的背景图resultImage.jpg来求得并标注轨道区域