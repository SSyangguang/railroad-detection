# 使用线性拟合对铁轨区域进行绘制
# 输入为视频预处理过程中求得的背景图，处理完成后会分别保存绘制了铁轨区域的原图和只有铁轨区域的图像

import numpy as np
import cv2

# 定义边缘检测中需要用到的参数
blurKernel = 21  # Gaussian blur kernel size
cannyLowThreshold = 10  # Canny edge detection low threshold
cannyHighThreshold = 130  # Canny edge detection high threshold

# 定义hough变换参数
rho = 1     # rho的步长，即直线到图像原点(0,0)点的距离
theta = np.pi / 180     # theta的范围
threshold = 50      # 累加器中的值高于它时才认为是一条直线
min_line_length = 150   # 线的最短长度，小于该值的线段会被忽略
max_line_gap = 20   # 两条直线之间的最大间隔，小于此值，认为是一条直线

def roi_mask(img, vertices):
    # img是输入的图像，vertices是ROI四个点的坐标
    # 生成与输入图像分辨率相同的纯黑图像，用于后期绘制铁轨区域
    mask = np.zeros_like(img)  
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    # 判断图像通道数量，如果不是单通道图像，则为每个图像的每个通道都添加白色蒙版
    if len(img.shape) > 2:  
        # 图像的通道数量
        channel_count = img.shape[2]  # 将图像的通道数量赋值给channel_count
        mask_color = (255,) * channel_count  # 例如三通道彩图channel_count=3,那么mask_color=(255, 255, 255)
    else:
        mask_color = 255    # 单通道灰度图
    cv2.fillPoly(mask, vertices, mask_color)  # 使用白色填充多边形，形成蒙板
    masked_img = cv2.bitwise_and(img, mask)  # 使用蒙版与原图相与，得到只有ROI区域的图像
    
    return masked_img



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, ymin, ymax):
    # 函数输出的直接就是一组直线点的坐标位置（每条直线用两个点表示[x1,y1],[x2,y2]）
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    draw_lanes(line_img, lines, ymin, ymax)

    return line_img

def draw_lanes(img, lines, ymin, ymax):
    # 初始化存储两条铁轨直线的坐标
    left_lines, right_lines = [], []
    # 通过铁轨斜率，将该直线存入left_lines或者right_lines，斜率需要自己调整
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k > 1:
                left_lines.append(line)
            else:
                right_lines.append(line)

    # clean_lines(left_lines, 0.1)  # 弹出左侧不满足斜率要求的直线
    # clean_lines(right_lines, 0.1)  #弹出右侧不满足斜率要求的直线

    # 提取左侧直线族中的所有的第一个点
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    # 提取左侧直线族中的所有的第二个点
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    # 提取右侧直线族中的所有的第一个点
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    # 提取右侧侧直线族中的所有的第二个点
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    # 拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标
    left_vtx = calc_lane_vertices(left_points, ymin, ymax)
    # 拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标
    right_vtx = calc_lane_vertices(right_points, ymin, ymax)

    # 初始化铁轨区域多边形四个顶点的坐标
    vtx = []

    print(left_vtx[0])
    print(left_vtx[1])
    print(right_vtx[1])
    print(right_vtx[0])

    # 依次添加左下角、左上角、右上角、右下角的坐标
    vtx.append(left_vtx[0])
    vtx.append(left_vtx[1])
    vtx.append(right_vtx[1])
    vtx.append(right_vtx[0])

    # cv2.fillPoly()中传入的坐标点需要是三维的，所以需要这步再添加一个维度
    vtx = np.array([vtx])

    # 通过坐标点，填充铁轨区域为(0, 255, 0)
    cv2.fillPoly(img, vtx, (0, 255, 0))

# 将不满足斜率要求的直线弹出
def clean_lines(lines, threshold):
    slope = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            slope.append(k)

    while len(lines) > 0:
        # 计算斜率的平均值，因为后面会将直线和斜率值弹出
        mean = np.mean(slope)
        # 计算每条直线斜率与平均值的差值
        diff = [abs(s - mean) for s in slope]
        # 计算差值的最大值的下标
        idx = np.argmax(diff)
        if diff[idx] > threshold:   # 将差值大于阈值的直线弹出
          slope.pop(idx)    # 弹出斜率
          lines.pop(idx)    # 弹出直线
        else:
          break

# 拟合点集，生成直线表达式，并计算直线在图像中的两个端点的坐标
def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]  # 提取x
    y = [p[1] for p in point_list]  # 提取y
    fit = np.polyfit(y, x, 1)  # 用一次多项式x=a*y+b拟合这些点，fit是(a,b)
    fit_fn = np.poly1d(fit)  # 生成多项式对象a*y+b

    xmin = int(fit_fn(ymin))  # 计算这条直线在图像中最左侧的横坐标
    xmax = int(fit_fn(ymax))  # 计算这条直线在图像中最右侧的横坐标

    return [(xmin, ymin), (xmax, ymax)]

def processing(img):
    # 定义ROI区域的四个顶点，分别对应左下角、左上角、右上角和右下角四个点的坐标
    roi_vtx = np.array([[(810, 1080), (810, 490), (1920, 490), (1920, 1080)]])

    # ymin和ymax是后期规定铁轨区域时限制的范围，即ROI的上下边界Y轴坐标
    ymin = roi_vtx[0][1][1]
    ymax = roi_vtx[0][0][1]

    # 输入图像预处理，对图像进行高斯滤波和canny边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blurKernel, blurKernel), 0, 0)
    edges = cv2.Canny(blur_gray, cannyLowThreshold, cannyHighThreshold)

    # 根据输入的ROI区域顶点坐标，将输入图像除了ROI区域像素置为0，相当于给图像ROI添加蒙版
    roi_edges = roi_mask(edges, roi_vtx)

    # 使用hough变换检测直线，并将检测出来的直线进行筛选后绘制出铁轨区域
    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap, ymin, ymax)
  
    # 将铁轨区域和输入图像进行叠加
    res_img = cv2.addWeighted(img, 1, line_img, 0.5, 0)

    return res_img, line_img



