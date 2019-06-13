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
2019/4/29

This file is for filtering the RGB images

'''


res='/home/zhangdongyu/object-detection/detectron/data/VOC2007/JPEGImages_cp'
#res='/home/zhangdongyu/object-detection/detectron/data/VOC2007/paste_output'
test_list ='/home/zhangdongyu/object-detection/detectron/data/VOC2007/ImageSets/Main/train.txt'
parsedLabel='/home/zhangdongyu/object-detection/detectron/data/VOC2007/txt'
savePath = '/home/zhangdongyu/object-detection/detectron/data/VOC2007/paste_output'

object_class = ['ignored_regions','car','truck','bus']

if not os.path.exists(savePath):
   os.mkdir(savePath)

def paste_regions(im,p,filename,X,Y):
    
    bb_list = open(filename,'r')
    bb_all = bb_list.readlines()
    for bb in bb_all[:]:
        box = bb.strip().split(",")
        x = int(box[0]) 
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        #region = im[y:y+h, x:x+w]
        print(x,y,x+w-1,y+h-1)
        add = 25
        add1 = 15
        add2 = 5
        if (x-add>0) and (y-add>0) and (x+w+add<X) and (y+h+add<Y):
             region = im.crop((x-add,y-add,x+w+add,y+h+add))
             p.paste(region, (x-add, y-add))
        elif (x-add1 > 0) and (y-add1 > 0) and (x+w+add1<X) and (y+h+add1<Y):
             region = im.crop((x-add1,y-add1,x+w+add1,y+h+add1))
             p.paste(region, (x-add1, y-add1))
        elif (x-add2 > 0) and (y-add2 > 0) and (x+w+add2<X) and (y+h+add2<Y):
             region = im.crop((x-add2,y-add2,x+w+add2,y+h+add2))
             p.paste(region, (x-add2, y-add2))
        else:
             region = im.crop((x,y,x+w-1,y+h-1))
             p.paste(region, (x, y) )
        p.save(savePath+"/"+name+".jpg")

    return 1


f = open(test_list,'r')
count = 0
lines = f.readlines()

for line in lines[:]:
	name = line.strip()
	print (name)
	im = Image.open(res+"/"+name+".jpg")
	X,Y = im.size
	#h = im.shape[0]
	#d = im.shape[2]
	print (X,Y)
        #创建白色底照
	p = Image.new('RGB', (X,Y), (255,255,255))
	filename = parsedLabel+"/"+name+".txt"
	paste_regions(im,p,filename,X,Y)
