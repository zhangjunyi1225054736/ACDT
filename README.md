
# Introduction
This is the codes of ICCV workshop paper 《How to Fully Exploit The Abilities of Aerial Image Detectors》.
Our model is based on detectron.pytorch.

![image](https://github.com/zhangjunyi1225054736/ACDT/blob/master/Selection_294.png)


# Environment
The environment required is exactly the same as https://github.com/roytseng-tw/Detectron.pytorch
# Datasets
1、Visdrone http://www.aiskyeye.com/upfile/Vision_Meets_Drones_A_Challenge.pdf
2、UAVDT https://sites.google.com/site/daviddo0323/projects/uavdt

# Backbone
We choose e2e_mask_rcnn_R-101-FPN_1x.yaml and e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.

# How to train
bash zjy_train.sh

# How to test
bash zjy_test.sh

# Result

The results in dataset Visdrone:

![image](https://github.com/zhangjunyi1225054736/ACDT/blob/master/Selection_292.png)

The results in dataset UAVDT:

![image](https://github.com/zhangjunyi1225054736/ACDT/blob/master/Selection_293.png)

