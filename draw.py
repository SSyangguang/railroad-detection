# 用于获取图像上某点的坐标，获取方法直接点击图像上需要的点即可

import cv2

# 这段程序用来获取鼠标在图中所点位置的坐标
img = cv2.imread('background_image.jpg')
a =[]
b = []
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    2.0, (0, 255, 0), thickness=2)
        cv2.imshow("image", img)

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)
print(a[0], b[0])

img[b[0]:b[1], a[0]:a[1], :] = 0
cv2.imshow("image", img)
cv2.waitKey(1000)
print (a, b)
