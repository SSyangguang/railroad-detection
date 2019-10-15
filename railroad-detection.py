# 通过hough变换，导入之前的背景图，在背景图中检测铁道位置
from PIL import Image, ImageEnhance
import numpy as np
import cv2

# Canny检测
def do_canny(frame):
	# 将每一帧转化为灰度图像，去除多余信息
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# 高斯滤波
	blur = cv2.GaussianBlur(frame, (5, 5), 0)
	# 边缘检测， cv2.Canny(img, low_threshold, high_threshold)
	canny = cv2.Canny(blur, 50, 150)

	return canny

def do_sobel(frame):
	# sobel算子检测图像边缘
	# x,y方向分别进行计算
	x = cv2.Sobel(frame, cv2.CV_16S, 1, 0)
	y = cv2.Sobel(frame, cv2.CV_16S, 0, 1)

	# 转回uint8格式
	absX = cv2.convertScaleAbs(x)
	absY = cv2.convertScaleAbs(y)
	# 合并x和y轴方向的边缘
	dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
	# 显示图像
	cv2.imshow("Result", dst)
	cv2.imwrite("sobel-edge.jpg", dst)

	return dst



# 车道左右边界标定
def calculate_lines(frame, lines):
	# 建立两个空列表，用于存储左右车道边界坐标
	left = []
	right = []

	# 循环遍历lines
	for line in lines:
		# 将线段信息从二维转化能到一维
		x1,y1,x2,y2 = line.reshape(4)

		# 将一个线性多项式拟合到x和y坐标上，并返回一个描述斜率和y轴截距的系数向量
		parameters = np.polyfit((x1,x2), (y1,y2), 1)
		slope = parameters[0] #斜率
		y_intercept = parameters[1] #截距

		# 通过斜率大小，可以判断是左边界还是右边界
		# 很明显左边界slope<0(注意cv坐标系不同的)
		# 右边界slope>0
		if slope < 0:
			left.append((slope,y_intercept))
		else:
			right.append((slope,y_intercept))

	# 将所有左边界和右边界做平均，得到一条直线的斜率和截距
	left_avg = np.average(left,axis=0)
	right_avg = np.average(right,axis=0)
	# 将这个截距和斜率值转换为x1,y1,x2,y2
	left_line = calculate_coordinate(frame,parameters=left_avg)
	right_line = calculate_coordinate(frame, parameters=right_avg)

	return np.array([left_line,right_line])



# 将截距与斜率转换为cv空间坐标
def calculate_coordinate(frame, parameters):
	# 获取斜率与截距
	slope, y_intercept = parameters

	# 设置初始y坐标为自顶向下(框架底部)的高度
	# 将最终的y坐标设置为框架底部上方150
	y1 = frame.shape[0]
	y2 = int(y1-150)
	# 根据y1=kx1+b,y2=kx2+b求取x1,x2
	x1 = int((y1-y_intercept)/slope)
	x2 = int((y2-y_intercept)/slope)
	return np.array([x1,y1,x2,y2])



# 可视化车道线
def visualize_lines(frame,lines):
	lines_visualize = np.zeros_like(frame)
	# 检测lines是否为空
	if lines is not None:
		for x1,y1,x2,y2 in lines:
			# 画线
			cv2.line(lines_visualize,(x1,y1),(x2,y2),(0,0,255),5)
	return lines_visualize


def img_processing(img):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
	# canny边缘检测
	# edges = cv2.Canny(binary, 50, 150, apertureSize=3)

	edges = do_canny(binary)
	return edges


def line_detect(img):
	# 该函数使用hough变换检测出直线并在原始图像中标注出来
	result = img_processing(img)

	# 使用hough变换进行线检测

	lines = cv2.HoughLinesP(result, 1, 1 * np.pi / 180, 10, minLineLength=10, maxLineGap=5)
	# print(lines)
	print("Line Num : ", len(lines))

	# 画出检测的线段
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
		pass
	img = Image.fromarray(img, 'RGB')
	img.show()


if __name__ == "__main__":
    background = cv2.imread('resultImage.jpg')
    cv2.imshow('original image', background)

    # # 使用canny算子检测边缘
    # cannyImage = do_canny(background)
    # # 使用hough变换检测直线
    # # hough变换代码：
    # # cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)
    # # 返回的是线条两个断点的值
    # houghImage = cv2.HoughLinesP(cannyImage, 2, np.pi/180, 100,
	# 		     minLineLength=100, maxLineGap=50)
    #
    #
    # # 将从hough检测到的多条线平均成一条线表示车道的左边界，
    # # 一条线表示车道的右边界
    # lines = calculate_lines(background, houghImage)
    #
    # # 可视化
    # lines_visualize = visualize_lines(background, lines)  # 显示
    # cv2.imshow("lines", lines_visualize)
    # # 叠加检测的车道线与原始图像,配置两张图片的权重值
    # # alpha=0.6, beta=1, gamma=1
    # output = cv2.addWeighted(background, 0.6, lines_visualize, 0.4, 0)
    # # output = line_detect(background)

    output = line_detect(background)
    cv2.imshow("output", output)

    cv2.waitKey(0)