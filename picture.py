import os
import cv2
import cv2.cv as cv
import sys
import socket
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import segmentation,measure

HOST = '192.168.1.166'
PORT = 10006
strLight = ""
def detect(filepath, file):

    font = cv2.FONT_HERSHEY_SIMPLEX   #设置显示字体类型，正常大小无衬线字体
    img = cv2.imread(filepath+file)
    cimg = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#BGR转HSV

    # color range
    lower_red1 = np.array([0,100,100])        #定义在HSV空间中红色的范围
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])      #定义在HSV空间中红色的范围
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])        #定义在HSV空间中绿色的范围
    upper_green = np.array([90,255,255])
    lower_yellow = np.array([15,150,150])     #定义在HSV空间中黄色的范围
    upper_yellow = np.array([35,255,255])
    #根据以上定义的红绿黄颜色方位得到个颜色交通灯部分
    mask1 = cv2.inRange(hsv, lower_red1,   upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2,   upper_red2)
    mask_g = cv2.inRange(hsv, lower_green,  upper_green)
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_r = cv2.add(mask1, mask2)

    #定义结构元素，形态学开运算
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1))#定义MORPH_CROSS比MORPH_RECT效果好
    #开运算
    opened_r  = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, element)
    opened_g  = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, element)
    opened_y  = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, element)

    ################检测红色交通灯########################
    segmentation.clear_border(opened_r)  #清除与边界相连的目标物
    label_image_r = measure.label(opened_r)  #连通区域标记
    borders_r = np.logical_xor(mask_r, opened_r) #异或
    label_image_r[borders_r] = -1
    for region_r in measure.regionprops(label_image_r): #循环得到每一个连通区域属性集   
        #忽略小区域和大面积	
        if region_r.convex_area < 120 or region_r.area > 2000:
            continue
        #绘制外包矩形
        area = region_r.area                   #连通域面积,区域内像素点总数
        eccentricity = region_r.eccentricity   #连通域离心率
        convex_area  = region_r.convex_area    #凸包内像素总数
        minr, minc, maxr, maxc = region_r.bbox #边界外连接
        radius = max(maxr-minr,maxc-minc)/2    #连通域外接矩形长轴的一半
        centroid = region_r.centroid           #质心坐标
        perimeter    = region_r.perimeter      #区域周长
        x = int(centroid[0])
        y = int(centroid[1])
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)

        if perimeter == 0:
            circularity = 1
        else:
            circularity = 4*3.141592*area/(perimeter*perimeter)
            circum_circularity      = 4*3.141592*convex_area/(4*3.1592*3.1592*radius*radius) 

        if eccentricity <= 0.4 or circularity >= 0.7 or circum_circularity >= 0.73:
            cv2.circle(cimg, (y,x), radius, (0,0,255),3)
            cv2.putText(cimg,'RED',(y,x), font, 1,(0,0,255),2)
            return "RED",cimg
        else:
            continue

    ################检测绿色交通灯########################
    segmentation.clear_border(opened_g)  #清除与边界相连的目标物
    label_image_g = measure.label(opened_g)  #连通区域标记
    borders_g = np.logical_xor(mask_g, opened_g) #异或
    label_image_g[borders_g] = -1
    #image_label_overlay_g = color.label2rgb(label_image_g, image=opened_g) #不同标记用不同颜色显示
    for region_g in measure.regionprops(label_image_g): #循环得到每一个连通区域属性集
        if region_g.convex_area < 130 or region_g.area > 2000:
            continue
        area = region_g.area   #连通域面积
        eccentricity = region_g.eccentricity   #连通域离心率
        convex_area  = region_g.convex_area    #凸包内像素总数
        minr, minc, maxr, maxc = region_g.bbox #边界外连接
        radius       = max(maxr-minr,maxc-minc)/2    #连通域外接矩形长轴的一半
        centroid     = region_g.centroid           #质心坐标
        perimeter    = region_g.perimeter      #区域周长
        x = int(centroid[0])
        y = int(centroid[1])
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)

        if perimeter == 0:
            circularity = 1
        else:
            circularity = 4*3.141592*area/(perimeter*perimeter)
            circum_circularity      = 4*3.141592*convex_area/(4*3.1592*3.1592*radius*radius) 

        if eccentricity <= 0.4 or circularity >= 0.7 or circum_circularity >= 0.8:
            cv2.circle(cimg, (y,x), radius, (0,255,0),3)
            cv2.putText(cimg,'GREEN',(y,x), font, 1,(0,255,0),2)
            return "GREEN",cimg
        else:
            continue
##
##    ################检测黄色交通灯########################
##    segmentation.clear_border(opened_y)  #清除与边界相连的目标物
##    label_image_y = measure.label(opened_y)  #连通区域标记
##    borders_y = np.logical_xor(mask_y, opened_y) #异或
##    label_image_y[borders_y] = -1
##    #image_label_overlay_y = color.label2rgb(label_image_y, image=opened_y) #不同标记用不同颜色显示
##    for region_y in measure.regionprops(label_image_y): #循环得到每一个连通区域属性集
##        if region_y.convex_area < 130 or region_y.area > 2000:
##            continue
##        area = region_y.area   #连通域面积
##        eccentricity = region_y.eccentricity   #连通域离心率
##        convex_area  = region_y.convex_area    #凸包内像素总数
##        minr, minc, maxr, maxc = region_y.bbox #边界外连接
##        radius       = max(maxr-minr,maxc-minc)/2    #连通域外接矩形长轴的一半
##        centroid     = region_y.centroid           #质心坐标
##        perimeter    = region_y.perimeter      #区域周长
##        x = int(centroid[0])
##        y = int(centroid[1])
##        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
##                                  fill=False, edgecolor='red', linewidth=2)
##
##        if perimeter == 0:
##            circularity = 1
##        else:
##            circularity = 4*3.141592*area/(perimeter*perimeter)
##            circum_circularity      = 4*3.141592*convex_area/(4*3.1592*3.1592*radius*radius) 
##
##        if eccentricity <= 0.4 or circularity >= 0.7 or circum_circularity >= 0.8:
##            cv2.circle(cimg, (y,x), radius, (0,255,255),3)
##            cv2.putText(cimg,'YELLOW',(y,x), font, 1,(0,255,255),2)
##            return "YELLOW",cimg
##        else:
##            continue

    return "NONE",img

def max(a, b):
    if a>b:
        return a
    else: 
        return b

if __name__ == '__main__':
    #作为客户端，请求连接服务器端
    #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #print("socket crerated...")
    #sock.connect((HOST, PORT))
    #print("socket connect complete...")

    #打开保存图像的文件夹，读取图片
    #path = os.path.abspath('..\\..')+'\\PythonTLR\\0403\\'
    path = 'E:\\2016UVC\\Traffic Light Recognition\\PythonTLR\\camera1023\\'
    redLight = 0
    greenLight = 0
    yellowLight = 0
    allLight = 0 
    videoWriter = cv2.VideoWriter("save.avi", cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 20.0, (1200,800))
    for f in os.listdir(path):
        if f.endswith('.bmp') or f.endswith('.jpg'):
            strLight,result = detect(path, f)
            cv2.imshow("result", result)
            cv2.waitKey(1000)
            videoWriter.write(result) #写视频帧
        if strLight == "NONE":
            continue
        elif str(strLight) == "RED":
            redLight = redLight + 1
            allLight = allLight + 1
            print("detected the light: " + f + " "+str(strLight))
            if allLight >=20:
                if (redLight/allLight) >= 0.9:
                    #cmd = raw_input("0xff 01")     ##如果检测到红灯，发送消息
                    #sock.sendall(cmd)
                    redLight = 0
                    allLight = 0                     
        elif str(strLight) == "GREEN":
            greenLight = greenLight + 1
            allLight = allLight + 1
            print("detected the light: " + str(strLight))
            if allLight >= 20:
                if (greenLight/allLight) >= 0.9:
                    #cmd = raw_input("0xff 02")     ##如果检测到绿灯，发送消息
                    #sock.sendall(cmd)
                    greenLight = 0
                    allLight = 0
        elif str(strLight) == "YELLOW":
            yellowLight = yellowLight + 1
            allLight = allLight + 1
            print("detected the light: " + str(strLight))
            if allLight >= 20:
                if (yellowLight/allLight) >= 0.9:
                    #cmd = raw_input("0xff 03")     ##如果检测到黄灯，发送消息
                    #sock.sendall(cmd)
                    yellowLight = 0
                    alllight = 0
    #sock.close()        #关闭socket通信
    #del(capture) 
