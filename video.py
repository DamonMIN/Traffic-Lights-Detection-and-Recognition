import os
import cv2
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
def detect(frame):

    font = cv2.FONT_HERSHEY_SIMPLEX   #������ʾ�������ͣ�������С�޳�������
    cimg = frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#BGRתHSV

    # color range
    lower_red1 = np.array([0,100,100])        #������HSV�ռ��к�ɫ�ķ�Χ
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])      #������HSV�ռ��к�ɫ�ķ�Χ
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])        #������HSV�ռ�����ɫ�ķ�Χ
    upper_green = np.array([90,255,255])
    lower_yellow = np.array([15,150,150])     #������HSV�ռ��л�ɫ�ķ�Χ
    upper_yellow = np.array([35,255,255])
	#�������϶���ĺ��̻���ɫ��λ�õ�����ɫ��ͨ�Ʋ���
    mask1 = cv2.inRange(hsv, lower_red1,   upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2,   upper_red2)
    mask_g = cv2.inRange(hsv, lower_green,  upper_green)
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_r = cv2.add(mask1, mask2)

    #����ṹԪ�أ���̬ѧ������
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1))#����MORPH_CROSS��MORPH_RECTЧ����
    #������
    opened_r  = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, element)
    opened_g  = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, element)
    opened_y  = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, element)

    ################����ɫ��ͨ��########################
    segmentation.clear_border(opened_r)           #�����߽�������Ŀ����
    label_image_r = measure.label(opened_r)       #��ͨ������
    borders_r = np.logical_xor(mask_r, opened_r)  #���
    label_image_r[borders_r] = -1
    for region_r in measure.regionprops(label_image_r): #ѭ���õ�ÿһ����ͨ�������Լ�   
        #����С����ʹ����	
        if region_r.convex_area < 120 or region_r.area > 2000:
            continue
        #�����������
        area = region_r.area                      #��ͨ�����,���������ص�����
        eccentricity = region_r.eccentricity      #��ͨ��������
        convex_area  = region_r.convex_area       #͹������������
        minr, minc, maxr, maxc = region_r.bbox    #�߽�������
        perimeter    = region_r.perimeter         #�����ܳ�
        radius = max(maxr-minr,maxc-minc)/2       #��ͨ����Ӿ��γ����һ��
        centroid = region_r.centroid              #��������
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

    ################�����ɫ��ͨ��########################
    segmentation.clear_border(opened_g)            #�����߽�������Ŀ����
    label_image_g = measure.label(opened_g)        #��ͨ������
    borders_g = np.logical_xor(mask_g, opened_g)   #���
    label_image_g[borders_g] = -1
    for region_g in measure.regionprops(label_image_g): #ѭ���õ�ÿһ����ͨ�������Լ�
        if region_g.convex_area < 130 or region_g.area > 2000:
            continue
        area = region_g.area                       #��ͨ�����
        eccentricity = region_g.eccentricity       #��ͨ��������
        convex_area  = region_g.convex_area        #͹������������
        minr, minc, maxr, maxc = region_g.bbox     #�߽�������
        radius       = max(maxr-minr,maxc-minc)/2  #��ͨ����Ӿ��γ����һ��
        centroid     = region_g.centroid           #��������
        perimeter    = region_g.perimeter          #�����ܳ�
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

    ################����ɫ��ͨ��########################
    segmentation.clear_border(opened_y)           #�����߽�������Ŀ����
    label_image_y = measure.label(opened_y)       #��ͨ������
    borders_y = np.logical_xor(mask_y, opened_y)  #���
    label_image_y[borders_y] = -1
    for region_y in measure.regionprops(label_image_y): #ѭ���õ�ÿһ����ͨ�������Լ�
        if region_y.convex_area < 130 or region_y.area > 2000:
            continue
        area = region_y.area   #��ͨ�����
        eccentricity = region_y.eccentricity      #��ͨ��������
        convex_area  = region_y.convex_area       #͹������������
        minr, minc, maxr, maxc = region_y.bbox    #�߽�������
        radius       = max(maxr-minr,maxc-minc)/2 #��ͨ����Ӿ��γ����һ��
        centroid     = region_y.centroid          #��������
        perimeter    = region_y.perimeter         #�����ܳ�
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
            cv2.circle(cimg, (y,x), radius, (0,255,255),3)
            cv2.putText(cimg,'YELLOW',(y,x), font, 1,(0,255,255),2)
            return "YELLOW",cimg
        else:
            continue
    return "NONE",frame


def max(a, b):
    if a>b:
        return a
    else: 
        return b

if __name__ == '__main__':
    
     #��Ϊ�ͻ��ˣ����ӷ�����
     #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
     #print("socket crerated...")
     #sock.connect((HOST, PORT))
     #print("socket connect complete...")

     #������ͷ��ʵʱ����
     redLight = 0
     greenLight = 0
     yellowLight = 0
     allLight = 0 
     videoCapture = cv2.VideoCapture(0)   #������ͷ
     frameIndex = 0                  #֡ͼ������
     #������ʼ��ߴ�
     fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
     size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
     videoWriter = cv2.VideoWriter("save.avi", cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 25.0, size)
     if not videoCapture.isOpened():
         print("can't open the camera!")
         sys.exit()                  #�˳�ϵͳ
     else:
         print("open the camera successfully!")
         while (1):
             bGrabbed, frame = videoCapture.read()
             if bGrabbed:
                 strLight,result = detect(frame)
                 videoWriter.write(result) #д��Ƶ֡  
             if strLight == "NONE":
                 print("No lights in scene!")
                 continue
             elif str(strLight) == "RED":
                 redLight = redLight + 1
                 allLight = allLight + 1
                 print("detected the light: " + str(strLight))
                 if allLight >=20:
                     if (redLight/allLight) >= 0.9:
                         #cmd = raw_input("0xff 01")     ##�����⵽��ƣ�������Ϣ
                         #sock.sendall(cmd)
                         redLight = 0
                         allLight = 0
             elif str(strLight) == "GREEN":
                 greenLight = greenLight + 1
                 allLight = allLight + 1
                 print("detected the light: " + str(strLight))
                 if allLight >= 20:
                     if (greenLight/allLight) >= 0.9:
                         #cmd = raw_input("0xff 02")     ##�����⵽�̵ƣ�������Ϣ
                         #sock.sendall(cmd)
                         greenLight = 0
                         allLight = 0
             elif str(strLight) == "YELLOW":
                 yellowLight = yellowLight + 1
                 allLight = allLight + 1
                 print("detected the light: " + str(strLight))
                 if allLight >= 20:
                     if (yellowLight/allLight) >= 0.9:
                         #cmd = raw_input("0xff 03")     ##�����⵽�Ƶƣ�������Ϣ
                         #sock.sendall(cmd)
                         yellowLight = 0
                         alllight = 0 
         cv2.destroyAllWindows()        
    #sock.close()        #�ر�socketͨ��
    #del(capture) 
