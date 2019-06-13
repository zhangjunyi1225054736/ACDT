import os
import csv

path = '/home/zhangdongyu/object-detection/detectron/data/VOC2007/txt/'
files = os.listdir(path)

a = []

for file in files:
	temp= [file[0:23],0,0,0,0,0,0,0,0,0,0,0,0,0]
	with open(path+file, 'r') as f:
		for line in f:
			data = line.strip().split(',')
			temp[int(data[5])+1] += 1
			temp[13] += 1
	f.close()
	temp[13] -= (temp[1]+temp[12])
	data = []
	data.append(temp[0])
	for i in range(1,12):
		if i!=1 and i!=12:	
			data.append(temp[i]/ temp[13])
	data.append(temp[13])
	a.append(data)
'''
with open('statistic.txt', 'w+') as f:
    for i in range(len(a)):
    	f.writelines(str(a[i])+'\n')
f.close()
'''
out = open("statistic.txt","w", newline='')
csv_write = csv.writer(out)
csv_write.writerow(['id','pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor','total'])
for i in range(len(a)):
    csv_write.writerow(a[i])

