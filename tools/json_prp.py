#coding=utf-8

import numpy as np
import cv2
import json
import os 

# objects365_Tiny_Testset_images_list.json(5211)  objects365_Tiny_train.json(8688)  objects365_Tiny_val.json(622)  objects365_train.json(608606)  objects365_val.json(30000)

source = "/mnt/md126/zhangjunyi/drone-object-detection/VOC2007/cocoformatJson/voc_2007_test.json"


source_365 = "/mnt/md126/zhangjunyi/365-object-detection/objects365_json/objects365_Tiny_Testset_images_list.json"

#source_365 = "/mnt/md126/zhangjunyi/365-object-detection/objects365_json/objects365_Tiny_Testset_images_list.json"

with open(source_365, 'r') as f:
        data = json.load(f)

#image = data["images"]

#a = ['__background__']
#categories = data["categories"]
for i in data:
    #a.append(i["name"])
    print(i)


#print(data.keys()) #['images', 'type', 'annotations', 'categories']

#print(data["images"][0]) #{'file_name': '0000244_02000_d_0000005.jpg', 'height': 540, 'width': 960, 'id': 0}

#print(data["annotations"][0]) #{'segmentation': [[47, 465, 47, 515, 111, 515, 111, 465]], 'area': 3200, 'iscrowd': 0, 'image_id': 0, 'bbox': [47, 465, 64, 50], 'category_id': 5, 'id': 1, 'ignore': 0}

#print(data["categories"])


#with open(source_365, 'r') as f:
        #data = json.load(f)

#print(data.keys()) #['categories', 'type', 'images', 'licenses', 'info', 'annotations']
#print(data["images"][0]) {'width': 480, 'height': 640, 'file_name': 'obj365_train_000000255476.jpg', 'id': 255476}
#print(data["annotations"][0]) #{'category_id': 301, 'bbox': [404.933288592, 430.874389632, 24.778381344000024, 20.832275392000042], 'iscrowd': 0, 'area': 516.1900639262046, 'image_id': 255476, 'id': 0}
#print(data["categories"])
