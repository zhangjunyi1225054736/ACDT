#coding=utf-8
from xml.dom import minidom,Node
import cv2
import string
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import os

'''
Created by Junyi Zhang
2019/5/13

'''


res='/home/zhangdongyu/object-detection/detectron/data/VOC2007/JPEGImages_cp'
test_list ='/home/zhangdongyu/object-detection/detectron/data/VOC2007/ImageSets/Main/train.txt'
parsedLabel='/home/zhangdongyu/object-detection/detectron/data/VOC2007/txt'
savePath = '/home/zhangdongyu/object-detection/detectron/data/VOC2007/output'

object_class = ['ignored_regions','car','truck','bus']

if not os.path.exists(savePath):
   os.mkdir(savePath)

def calculate_iou(a,b):
    a = np.array(a)

    box2 = np.array(b)#gt
    #a[:,0]<=box2[:,0]
    iou_list = []
    for box1 in a:
       #print("box1:",box1)
       #print("box2:",box2)
       
       count = ((box1[0]<=box2[:,0])&(box1[1]<=box2[:,1])&(box1[2]+box1[0]>=box2[:,2]+box2[:,0])&(box1[3]+box1[2]>=box2[:,3]+box2[:,2]))
       #print(count)
       #print(box1[0]<=box2[:,0])
       #print(box1[1]<=box2[:,1])
    
       iou_list.append(np.sum(count==True))
    return iou_list
'''
    box1 = a
    for box2 in b:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2
        #get the corrdinates of the intersection rectangle
        inter_rect_x1 =  max(b1_x1, b2_x1)
        inter_rect_y1 =  max(b1_y1, b2_y1)
        inter_rect_x2 =  min(b1_x2, b2_x2)
        inter_rect_y2 =  min(b1_y2, b2_y2)
        if b2_x1 >= b1_x1 and b2_y1>=b1_y1 and b2_x2<=b1_x2 and b2_y2<=b1_y2:
           iou = 1
        else:
           #Intersection area
           inter_width = inter_rect_x2 - inter_rect_x1 + 1
           inter_height = inter_rect_y2 - inter_rect_y1 + 1
           if inter_width > 0 and inter_height > 0:#strong condition
                inter_area = inter_width * inter_height
                #Union Area
                b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
                b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
                iou = inter_area / (b1_area + b2_area - inter_area)
           else:
                iou = 0
        #print(iou)
        if iou > 0.8:
    iou_list.append(iou) 
'''
    #return iou_list                    
   
def get_gt_list(name,parsedLabel):
    
    label_path = os.path.join(parsedLabel,name+'.txt')
    #print(file_path)
    label = open(label_path,'r')
    lines = label.readlines()
    box_list = np.zeros((len(lines),4))
    i = 0
    for line in lines[:]:
        line = line.strip().split(",")
        box = [int(line[0]),int(line[1]),int(line[2]),int(line[3])]
        box_list[i] = box
        i = i + 1
    return box_list

def count_score(a,gt_list): 
    #print(a)
    iou_list = calculate_iou(a,gt_list)
    return iou_list

def get_anchors(W,H):
    W = W -41
    H = H -41
    anchors = []
    count = 0 
    for i in range(0,W,10):
        for j in range(0,H,10):
           temp = []
           x = i
           y = j
           w = 40
           h = 40
           print(x,y,count)
           count = count + 1
           temp =[x,y,w,h]
           anchors.append(temp)            
    return anchors

f = open(test_list,'r')
count = 0
lines = f.readlines()

for line in lines[:]:
    name = line.strip()
    print (name)
    im = cv2.imread(res+"/"+name+".jpg")
    h = im.shape[0]
    w = im.shape[1]
    print (w,h)
    gt_list = get_gt_list(name,parsedLabel)
    a = get_anchors(w,h)
    print(len(a))
    #exit()
    score = count_score(a, gt_list)
    print("score:",max(score))
    index = score.index(max(score))
    #print("index:",index)
    print("box:",a[index])
    exit() 
