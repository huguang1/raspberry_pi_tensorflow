# import cv2
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2 as cv
# from PIL import Image
# import pytesseract as tess
# #验证码识别
# img = cv2.imread('pictures/0a2be55fd09e12ee6ca33b729fcd7d5621b553fe.png')
# img = cv2.blur(img, (3, 3))
# cv2.imshow('yzm', img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_TRIANGLE)
# kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
# open1 = cv2.erode(binary, kernel=kernel1)
# open2 = cv2.morphologyEx(open1, cv2.MORPH_OPEN, kernel=kernel2)
# # cv2.imshow('detect',open2)
#
# cv2.bitwise_not(open2, open2)
# textimg = Image.fromarray(open2)
# text = tess.image_to_string(textimg)
# print("验证码为:%s"%text)
# # cv2.waitKey()


# # -*- coding=GBK -*-
# import cv2 as cv
# from PIL import Image
# import pytesseract
#
#
# def recognize_text():
#     gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#     ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
#     cv.imshow("Binarization", binary)
#
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 8))
#     binl = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
#     cv.imshow("MORPH_OPEN_1", binl)
#
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (8, 1))
#     open_out = cv.morphologyEx(binl, cv.MORPH_OPEN, kernel)
#     cv.imshow("MORPH_OPEN_2", open_out)
#
#     cv.bitwise_not(open_out, open_out)  # 背景变为白色
#     cv.imshow("Transform", open_out)
#     textImage = Image.fromarray(open_out)
#     text = pytesseract.image_to_string(textImage)  # 验证码显示数字或者字母尽量并排，且transform中的字符须保持独立、无断开、完整连接的整体
#     print(text)
#     print("This OK: %s" % text)
#
#
# src = cv.imread("pictures/0a4cf35aeef3f171efe9b36743b74ed5c423e859.png")
# cv.imshow("before", src)
# recognize_text()
# cv.waitKey(0)
# cv.destroyAllWindows()


# coding:utf8
import numpy as np
import cv2
from PIL import Image
import pytesseract
import sys
# 读取文件
imagePath = "pictures/0a4cf35aeef3f171efe9b36743b74ed5c423e859.png"
img = cv2.imread(imagePath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img',gray)
# cv2.waitKey()
# 此步骤形态学变换的预处理，得到可以查找矩形的图片
# 参数：输入矩阵、输出矩阵数据类型、设置1、0时差分方向为水平方向的核卷积，设置0、1为垂直方向,ksize：核的尺寸
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
# cv2.imshow('sobel',sobel)
# cv2.waitKey()
# 二值化
ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
# 设置膨胀和腐蚀操作的核函数
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

# 膨胀一次，让轮廓突出
dilation = cv2.dilate(binary, element2, iterations = 1)
# cv2.imshow('dilation', dilation)
# cv2.waitKey()
# 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
erosion = cv2.erode(dilation, element1, iterations = 1)
# cv2.imshow('erosion', erosion)
# cv2.waitKey()
# aim = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,element1, 1 )   #此函数可实现闭运算和开运算
# 以上膨胀+腐蚀称为闭运算，具有填充白色区域细小黑色空洞、连接近邻物体的作用

# 再次膨胀，让轮廓明显一些
dilation2 = cv2.dilate(erosion, element2, iterations = 3)
# 显示膨胀一次后的图像处理效果
# plt.imshow(dilation2,'gray')
#
# # 显示一次闭运算后的效果
# plt.imshow(erosion,'gray')
#
# # 显示连续膨胀3次后的效果
# plt.imshow(dilation2,'gray')

#  查找和筛选文字区域
region = []
#  查找轮廓
contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 利用以上函数可以得到多个轮廓区域，存在一个列表中。
#  筛选那些面积小的
for i in range(len(contours)):
    # 遍历所有轮廓
    # cnt是一个点集
    cnt = contours[i]
    # 计算该轮廓的面积
    area = cv2.contourArea(cnt)
    # 面积小的都筛选掉、这个1000可以按照效果自行设置
    if(area < 1000):
        continue

#     # 将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定
#     # 轮廓近似，作用很小
#     # 计算轮廓长度
#     epsilon = 0.001 * cv2.arcLength(cnt, True)

#     #
# #     approx = cv2.approxPolyDP(cnt, epsilon, True)

    # 找到最小的矩形，该矩形可能有方向
    rect = cv2.minAreaRect(cnt)
    # 打印出各个矩形四个点的位置
    # print ("rect is: ")
    # print (rect)

    # box是四个点的坐标
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算高和宽
    height = abs(box[0][1] - box[2][1])
    width = abs(box[0][0] - box[2][0])

    # 筛选那些太细的矩形，留下扁的
    if(height > width * 1.3):
        continue
    region.append(box)
    # 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        print(box)
    # plt.imshow(img, 'brg')
    # plt.show()
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img: object = img[y1:y1 + hight, x1:x1 + width]
    print('result/%d'%i+'.jpg')
    # im=cv2.imshow('crop_img', crop_img)
    cv2.imwrite('result/%d'%i+'.jpg',crop_img,[int(cv2.IMWRITE_JPEG_QUALITY),70])
# 弹窗显示
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
    # 带轮廓的图片
 #Image = Image.open('1.png')   # 打开图片
# self.progressBar.setProperty("value", 0)
text = pytesseract.image_to_string(crop_img, lang='chi_sim')  #使用简体中文解析图片
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()




















