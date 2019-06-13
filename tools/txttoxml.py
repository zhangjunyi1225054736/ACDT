from xml.dom import minidom,Node
import cv2
import string
import os 

res='/home/zhangdongyu/object-detection/data/VOCdevkit2007/VOC2007/JPEGImages'
list ='/home/zhangdongyu/object-detection/data/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt'
parsedLabel='/home/zhangdongyu/object-detection/detectron/VOC2007/txt/'
savePath = '/home/zhangdongyu/object-detection/detectron/data/VOC2007/Annotations/'

if not os.path.exists(savePath):
 os.mkdir(savePath)

f = open(list,'r')
object_class = ['ignored_regions','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others']
#object_class = ['ignored_regions','car','truck','bus']
count = 0
lines = f.readlines()
for line in lines[:]:
	name = line.strip()
	print(name)
	im = cv2.imread(res+"/"+name+".jpg")
	w = im.shape[1]
	h = im.shape[0]
	d = im.shape[2]
	print (w,h,d)	
	doc = minidom.Document()
    
	annotation = doc.createElement('annotation')
	doc.appendChild(annotation)
	
	folder = doc.createElement('folder')
	folder.appendChild(doc.createTextNode("VOC2012")) 
	annotation.appendChild(folder)
	
	filename = doc.createElement('filename')
	filename.appendChild(doc.createTextNode(name)) 
	annotation.appendChild(filename)
	
	source = doc.createElement('source')
	database = doc.createElement('database')
	database.appendChild(doc.createTextNode("The VOC2012 Database")) 
	source.appendChild(database)
	annotation2 = doc.createElement('annotation')
	annotation2.appendChild(doc.createTextNode("UAV VOC2012")) 
	source.appendChild(annotation2)
	image = doc.createElement('image')
	image.appendChild(doc.createTextNode("image")) 
	source.appendChild(image)
	flickrid = doc.createElement('flickrid')
	flickrid.appendChild(doc.createTextNode("NULL")) 
	source.appendChild(flickrid)
	annotation.appendChild(source)
	
	owner = doc.createElement('owner')
	flickrid = doc.createElement('flickrid')
	flickrid.appendChild(doc.createTextNode("NULL")) 
	owner.appendChild(flickrid)
	na = doc.createElement('name')
	na.appendChild(doc.createTextNode("zhangjunyi")) 
	owner.appendChild(na)
	annotation.appendChild(owner)
	
	size = doc.createElement('size')
	width = doc.createElement('width')
	width.appendChild(doc.createTextNode("%d" %w)) 
	size.appendChild(width)
	height = doc.createElement('height')
	height.appendChild(doc.createTextNode("%d" %h)) 
	size.appendChild(height)
	depth = doc.createElement('depth')
	depth.appendChild(doc.createTextNode("%d" %d)) 
	size.appendChild(depth)
	annotation.appendChild(size)
	
	segmented = doc.createElement('segmented')
	segmented.appendChild(doc.createTextNode("0")) 
	annotation.appendChild(segmented)
	
	txtLabel = open(parsedLabel+name+'.txt','r')
	boxes = txtLabel.readlines()
	for box in boxes:
		box = box.strip().split(',')
		print (box)
                #print box[0]
		object = doc.createElement('object')
		nm = doc.createElement('name')
		nm.appendChild(doc.createTextNode(object_class[int(box[5])])) 
		object.appendChild(nm)
		pose = doc.createElement('pose')
		pose.appendChild(doc.createTextNode("undefined")) 
		object.appendChild(pose)
		truncated = doc.createElement('truncated')
		truncated.appendChild(doc.createTextNode(box[6])) 
		object.appendChild(truncated)
		difficult = doc.createElement('difficult')
		difficult.appendChild(doc.createTextNode("0")) 
		object.appendChild(difficult)
		bndbox = doc.createElement('bndbox')
		xmin = doc.createElement('xmin')
		xmin.appendChild(doc.createTextNode(box[0])) 
		bndbox.appendChild(xmin)
		ymin = doc.createElement('ymax')
                #if (int(box[3])>=0):
		ymin.appendChild(doc.createTextNode(str(int(box[1])+int(box[3])-1))) 
                #else:
		#ymin.appendChild(doc.createTextNode(str(int(box[1])+int(box[3])))) 
		bndbox.appendChild(ymin)
		xmax = doc.createElement('xmax')
                #if (int(box[2])>=0):
		xmax.appendChild(doc.createTextNode(str(int(box[0])+int(box[2])-1)))
                #else:
		#xmax.appendChild(doc.createTextNode(str(int(box[0])-int(box[2])))) 
		bndbox.appendChild(xmax)
		ymax = doc.createElement('ymin')
		ymax.appendChild(doc.createTextNode(box[1])) 
		bndbox.appendChild(ymax)
		object.appendChild(bndbox)
		annotation.appendChild(object)
	savefile = open(savePath+name+'.xml','wb')
	savefile.write(doc.toprettyxml(encoding='utf-8'))
	savefile.close()
	count += 1
	print (count)

