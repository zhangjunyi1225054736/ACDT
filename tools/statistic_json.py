'''
get the number of the images including some special categories only
'''
import json
import numpy as np

with open('/data/zhangjunyi/drone-object-detection/data/VOC2007/cocoformatJson/voc_2007_train_crop.json','r') as f:
    data = json.load(f)
class_num = [0,0,0,0,0,0,0,0,0,0]
total = 0
dic = {}
no_crop_list = []
for image in data['images']:
    if len(image['file_name'].split('_')) == 4:
        no_crop_list.append(image['id'])
        #continue
    dic[image['id']] = [0,0,0,0,0,0,0,0,0,0]

for annotations in data['annotations']:
    id = annotations['image_id']
    if id not in dic.keys():
       continue
    category = annotations['category_id']
    if category in range(2,12):
        dic[id][category - 2] += 1

for d in dic.keys():
    for i in range(10):
        if dic[d][i] != 0:
            class_num[i] += 1

class_num_cut = [0,0,0,0,0,0,0,0,0,0]
select = 0
total = 0
cut_id_list = []
for d in dic.keys():
    if d in no_crop_list:
        continue
    count = 0
    for i in range(10):
        if i == 0 or i == 1 or i == 3: #pedestrian 0 ; people 1 ; so on
        #if i == 0 or i == 1 or i == 3:
            count += 1
        elif dic[d][i] == 0 :
            count += 1
    if count == 10:
        cut_id_list.append(d)
        pass
    else:
        select += 1
        for j in range(10):
            if dic[d][j] != 0:
                class_num_cut[j] += 1
    total += 1
#rint(total)
#rint(class_num)
#rint(select)
#rint(class_num_cut)

class_num_ori = [0,0,0,0,0,0,0,0,0,0]
for d in dic.keys():
    if d in no_crop_list:
        for i in range(10):
            if dic[d][i] != 0:
                class_num_ori[i] += 1
print(class_num_ori)

class_num_cut = np.array(class_num_cut)
#print(class_num_cut)
miu = np.mean(class_num_cut)
sigma = np.var(class_num_cut)
print(sigma)
#print(len(dic.keys())) 
a = 0
for d in dic.keys():
   # print(class_num_cut)
    if d in cut_id_list or d in no_crop_list:
        continue
    temp_num = class_num_cut.copy()
    for i in range(10):
        if dic[d][i] != 0:
            temp_num[i] -= 1
    temp_sigma = np.var(temp_num)
    if temp_sigma < sigma:
        #print(temp_num)
        #print(temp_sigma)
        #print(" ")
        class_num_cut = temp_num
        sigma = temp_sigma
        a += 1
    #exit()
print(class_num_cut)
print(total-a)
