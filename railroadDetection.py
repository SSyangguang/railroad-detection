# 使用多项式拟合来对拟合铁轨边缘，用得到的多项式来绘制铁轨区域
# 可以绘制弯曲铁轨，根据弯曲程度来决定拟合的多项式的次数

import cv2
import numpy as np
import os
from PIL import Image
import time
import line




# get all image in the given directory persume that this directory only contain image files
# 用于读取文件夹中所有图片
def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname + '/' + img_name for img_name in img_names]
    imgs = [cv2.imread(path) for path in img_paths]
    return imgs



def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_select(img, channel='s', thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'h':
        channel = hls[:, :, 0]
    elif channel == 'l':
        channel = hls[:, :, 1]
    else:
        channel = hls[:, :, 2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:, :, 0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output


def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:, :, 2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output


def find_line(binary_warped):
    # Take a histogram of the bottom half of the image
    # 计算输入阈值图像的直方图
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 3)
    right_fit = np.polyfit(righty, rightx, 3)

    return left_fit, right_fit, left_lane_inds, right_lane_inds

# 通过多项式拟合来得到两条铁轨的参数
def find_line_by_previous(binary_warped, left_fit, right_fit):
    # 输入binary_warped是阈值化的图像
    nonzero = binary_warped.nonzero()  # 找出非零元素的位置，输出两个向量，第一个向量为非零元素的横坐标，第二个为元素的纵坐标
    nonzeroy = np.array(nonzero[0])  # 存储非零元素的y坐标到变量nonzeroy
    nonzerox = np.array(nonzero[1])  # 存储非零元素的x坐标到变量nonzerox
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 3) + left_fit[1] * (nonzeroy ** 2) +
                                   left_fit[2] * nonzeroy + left_fit[3] - margin)) & (
                                  nonzerox < (left_fit[0] * (nonzeroy ** 3) +
                                              left_fit[1] * (nonzeroy ** 2) + left_fit[2] * nonzeroy + left_fit[
                                                  3] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 3) + right_fit[1] * (nonzeroy ** 2) +
                                    right_fit[2] * nonzeroy + right_fit[3] - margin)) & (
                                   nonzerox < (right_fit[0] * (nonzeroy ** 3) +
                                               right_fit[1] * (nonzeroy ** 2) + right_fit[2] * nonzeroy + right_fit[
                                                   3] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 3)  # 进行多项式拟合，这里的次数是3次
    right_fit = np.polyfit(righty, rightx, 3)  # 多项式拟合返回的结果是多项式的系数
    return left_fit, right_fit, left_lane_inds, right_lane_inds

# 应该是用来得到的轨道区域进行外扩
def expand(img):
    image = img
    _, green, _ = cv2.split(image)
    s = np.sum(green, axis=1)
    a = range(1080)
    for i in reversed(a):
        if s[i] < 200:
            break
        for j in range(1920):  # min x
            if green[i][j] == 255:
                break
        for k in reversed(range(1920)):  # max x
            if green[i][k] == 255:
                break
        for l in range(int(s[i] / 255)):  # s[i]/255  the number
            image[i, j - l, 2] = 255
        for l in range(int(s[i] / 255)):
            image[i, k + l, 2] = 255

    return image


def draw_area(undist, binary_warped, Minv, left_fit, right_fit):
    # Generate x and y values for plotting
    # 以输入的二值图binary_warped的高为总长度，绘制以这个值为总长度的数组
    # 其实就是下面用拟合函数当作这个函数的x轴，left_fitx和right_fitx就是y轴
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # 这里就是用之前拟合得到的系数来得到方程，left_fit里面存的就是三次多项式的系数，ploty就相当于是横坐标，这样相乘就得到纵坐标
    # 分别是左边和右边铁轨的纵坐标
    left_fitx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
    right_fitx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]
    # 建一个空图，利用多项式拟合的结果参数，用来绘制区域
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)  # 空图的尺寸
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))  # 三个通道都用上面定义的warp_zero来填充

    # Recast the x and y points into usable format for cv2.fillPoly()
    # pts_left存储的是左边轨道对应的区域边缘的所有坐标点
    # pts_right存储的是右边轨道对应的区域边缘的所有坐标点
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # 沿着水平方向将数组连接起来，这是为了对多边形填充函数cv2.fillPoly输入进行的处理
    # pts存储的是轨道区域边缘所有点的坐标
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    # cv2.fillPoly在图像上的多边形进行填充
    # color_warp是需要用来画相关区域的图像，中间的参数是多边形的坐标，最后的参数是填充的颜色
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))   # np.int_表示默认的整数类型

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # 使用np.array使得图像变回图像形式，可以正常显示
    undist = np.array(undist)

    # color_warp是还在透视变换空间上绘制的区域，需要将绘制的区域进行透视逆变换，变回原始图像空间
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # cv2.imshow('new1', newwarp)
    # cv2.waitKey(1000)
    # newwarp = expand(newwarp)
    # cv2.imshow('new2', newwarp)
    # cv2.waitKey(1000)

    # Combine the result with the original image
    # indist是原始图像，newwarp是绘制的区域，将这两幅图进行叠加
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # result是绘制了区域颜色，并和原始图像进行叠加后的图像，newwarp是没有经过叠加，只有绘制区域的图像
    return result, newwarp


def calculate_curv_and_pos(binary_warped, left_fit, right_fit):
    # Define y-value where we want radius of curvature
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    curvature = ((left_curverad + right_curverad) / 2)
    # print(curvature)
    lane_width = np.absolute(leftx[719] - rightx[719])
    lane_xm_per_pix = 3.7 / lane_width
    veh_pos = (((leftx[719] + rightx[719]) * lane_xm_per_pix) / 2.)
    cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
    distance_from_center = cen_pos - veh_pos
    return curvature, distance_from_center


def draw_values(img, curvature, distance_from_center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature: %sm" % (round(curvature))

    if distance_from_center > 0:
        pos_flag = 'right'
    else:
        pos_flag = 'left'

    cv2.putText(img, radius_text, (100, 100), font, 1, (255, 255, 255), 2)
    # center_text = "Vehicle is %.3fm %s of center"%(abs(distance_from_center),pos_flag)
    # cv2.putText(img,center_text,(100,150), font, 1,(255,255,255),2)
    return img


def thresholding(img):
    #setting all sorts of thresholds
    x_thresh = abs_sobel_thresh(img, orient='x', thresh_min=90, thresh_max=280)
    mag_thresh1 = mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 170))
    dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = hls_select(img, thresh=(160, 255))
    lab_thresh = lab_select(img, thresh=(155, 210))
    luv_thresh = luv_select(img, thresh=(225, 255))

    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh1 == 1)) | ((dir_thresh == 1) &
                (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1

    return threshholded


# 返回的分别是绘制了区域的图像和ROI区域的二值图
def processing(img, M, Minv,left_line,right_line):

    prev_time = time.time() #记录当前时间

    # 对图像进行阈值化处理
    #get the thresholded binary image
    thresholded = thresholding(img)

    # 用来进行透视变换
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    # perform detection
    # left_line和right_line的类detected，
    if left_line.detected and right_line.detected:
        # 返回的left_fit和right_fit分别是左右两条轨道进行多项式拟合后得到的多项式的系数
        # 但似乎不知道left_lane_inds和right_lane_inds是什么参数，但是也没用到
        left_fit, right_fit, left_lane_inds, right_lane_inds = \
            find_line_by_previous(thresholded_wraped, left_line.current_fit, right_line.current_fit)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = find_line(thresholded_wraped)
    # 用拟合后得到的系数更新leftline和rightline
    left_line.update(left_fit)
    right_line.update(right_fit)
    # draw the detected laneline and the information
    undist = Image.fromarray(img)
    # Minv是透视变换的逆矩阵，因为最开始处理是先将轨道区域作透视变换为俯瞰形式处理，这里做完处理，得到多项式拟合参数后
    # 要再进行反变换，将俯瞰图还原为原始的图像形式

    # 通过拟合得到的多项式参数，在原图上在两个区域内绘制轨道的区域
    # area_img是叠加了区域的图像，gre1只是单独绘制了区域的图像
    area_img, gre1 = draw_area(undist, thresholded_wraped, Minv, left_fit, right_fit)
    # cv2.imshow('area_img', area_img)
    # cv2.waitKey(0)

    # 计算曲率信息
    curvature, pos_from_center = calculate_curv_and_pos(thresholded_wraped, left_fit, right_fit)

    area_img = np.array(area_img)

    # 将计算得到的曲率信息叠加到图像上
    result = draw_values(area_img, curvature, pos_from_center)

    curr_time = time.time()
    # 计算执行时间
    exec_time = curr_time - prev_time
    info = "time: %.2f ms" % (1000 * exec_time)
    print(info)
    return result, thresholded_wraped

# 正式程序开始的地方

# 初始化左右两条退铁轨的拟合参数
left_line = line.Line()
right_line = line.Line()


# 初始化透视变换的区域
src = np.float32([[(600, 1080), (850, 300), (1600, 1080), (1000, 300)]])
# 分别对应左下角、左上角、右下角、右上角,图中左上角为坐标原点，(横坐标,纵坐标)
dst = np.float32([[(500, 1080), (0, 0), (1500, 1080), (1300, 0)]])
# cv2.getPerspectiveTransform()用来求得透视变换的参数
# cv2.getPerspectiveTransform(src, dst) → retval 返回由源图像中矩形到目标图像矩形变换的矩阵
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
# M, Minv分别是src到dst和dst和src矩阵的变换，这里所定义的src和dst应该是作者自己测量的铁道区域的变换四个点的坐标


# pic文件夹中存储的是视频序列图片，将这些图片读取
img = cv2.imread('background_image.jpg')


prev_time = time.time()
# 返回的res是通过绘制区域后的图片，t1是透视变换后的图片，但是还是矩阵形式，所以不能以图片的形式显示出来
res, t1 = processing(img, M, Minv, left_line, right_line)
curr_time = time.time()
exec_time = curr_time - prev_time
info = "time: %.2f ms" % (1000 * exec_time)
print(info)
print(res.shape)
cv2.imwrite("result.jpg", res)
