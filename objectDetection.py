# 使用三帧差法获取运动目标
# 输入：代检测视频
# 输出：经过检测，对目标物体绘制矩形框后的视频

import cv2


# 读取视频文件
fileName = "videoCut.avi"
camera = cv2.VideoCapture(fileName)
if not camera.isOpened:
    print('Unable to open: ' + camera.input)
    exit(0)

# 获取视频信息，分别为视频的帧数和分辨率
fps = camera.get(cv2.CAP_PROP_FPS)
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 处理结果输出两个视频，一个是检测结果，一个是三帧差法求得的运动目标二值图
# 视频文件输出参数设置，帧数和尺寸与原始视频保持一致
out1 = cv2.VideoWriter('objectDetectionResult1.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)
# out2 = cv2.VideoWriter('objectDetectionResult2.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)


# 定义运动目标三帧差法检测函数
# 输入分别为三帧法的第一帧、第二帧、第三帧、检测完成视频保存的名字
def objectDetection(frame1, frame2, frame3, name):
    # 获取视频信息，分别为视频的帧数和分辨率
    fps = camera.get(cv2.CAP_PROP_FPS)
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 处理结果输出两个视频，一个是检测结果，一个是三帧差法求得的运动目标二值图
    # 视频文件输出参数设置，帧数和尺寸与原始视频保持一致
    out1 = cv2.VideoWriter('objectDetectionResult1.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    frameDelta1 = cv2.absdiff(frame1, frame2)
    # 计算当前帧和前帧的不同,计算三帧差分
    frameDelta2 = cv2.absdiff(frame2, frame3)  # 帧差二
    thresh = cv2.bitwise_and(frameDelta1, frameDelta2)  # 图像与运算

    # 结果转为灰度图
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    # 图像二值化
    thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]

    # 去除图像噪声,先腐蚀再膨胀(形态学开运算)
    thresh = cv2.dilate(thresh, None, iterations=3)
    thresh = cv2.erode(thresh, None, iterations=1)

    # 阀值图像上的轮廓位置
    # 将轮廓的边缘信息存储到cnts中
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    return cnts, hierarchy






# 初始化当前帧的前两帧
lastFrame1 = None
lastFrame2 = None

# 遍历视频的每一帧
while camera.isOpened():

    # 读取下一帧
    (ret, frame) = camera.read()
    # 如果不能抓取到一帧，说明我们到了视频的结尾
    if not ret:
        break

    # 调整该帧的大小
    # frame = cv2.resize(frame, (500, 400), interpolation=cv2.INTER_CUBIC)

    # 如果第一二帧是None，对其进行初始化,计算第一二帧的不同
    # frameDelta1是帧一和帧二的差
    if lastFrame2 is None:
        if lastFrame1 is None:
            lastFrame1 = frame
        else:
            lastFrame2 = frame
            # 声明全局变量frameDelta1，因为这是帧一和帧二的差值，须在整段函数中使用
            global frameDelta1  # 全局变量
            frameDelta1 = cv2.absdiff(lastFrame1, lastFrame2)  # 帧差一
        continue

    # 计算当前帧和前帧的不同,计算三帧差分
    frameDelta2 = cv2.absdiff(lastFrame2, frame)  # 帧差二
    thresh = cv2.bitwise_and(frameDelta1, frameDelta2)  # 图像与运算
    # 将三帧差的结果存入thresh2
    # thresh2 = thresh.copy()

    # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧,帧差二设为帧差一
    lastFrame1 = lastFrame2
    lastFrame2 = frame.copy()
    frameDelta1 = frameDelta2

    # 结果转为灰度图
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    # 图像二值化
    thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]

    # 去除图像噪声,先腐蚀再膨胀(形态学开运算)
    thresh = cv2.dilate(thresh, None, iterations=3)
    thresh = cv2.erode(thresh, None, iterations=1)

    # 阀值图像上的轮廓位置
    # 将轮廓的边缘信息存储到cnts中
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓
    for c in cnts:
        # 忽略小轮廓，排除误差
        if cv2.contourArea(c) < 300:
            continue

        # 计算轮廓的边界框，在当前帧中画出该框
        # 根据轮廓信息求矩形框，返回值是矩形框左上角的坐标和矩形框的长和高
        (x, y, w, h) = cv2.boundingRect(c)
        # 根据矩形框的坐标和尺寸，在当前帧图像上用绿色，在每个轮廓上绘制矩形框，绘制矩形框的线的宽度为2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示当前帧
    # cv2.imshow("frame", frame)
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("threst2", thresh2)

    # 保存视频
    out1.write(frame)
    # out2.write(thresh2)

    # 如果q键被按下，跳出循环
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

# 清理资源并关闭打开的窗口
out1.release()
# out2.release()
camera.release()
cv2.destroyAllWindows()