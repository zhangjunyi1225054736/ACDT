#!/bin/bash

#n is the number od test images
n=548
#step is the iteration step of the trained model
step=59999

cuda=6

rm -rf ./VOC2007/VOCdevkit2007/annotations_cache

#cp ./VOC2007/ImageSets/Main/test_$n.txt ./VOC2007/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
#cp ./VOC2007/ImageSets/Main/test_$n.txt ./VOC2007/ImageSets/Main/test.txt
 
#python ./tools/pascal_voc_xml2coco_json_converter.py ./data/ 2007

#CUDA_VISIBLE_DEVICES=$cuda python tools/test_net.py --dataset voc2007 --cfg configs/baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml \
         #--load_ckpt Outputs/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Apr28-14-53-17_zdyhpc_step/ckpt/model_step$step.pth --vis #--multi-gpu-testing

#CUDA_VISIBLE_DEVICES=$cuda python tools/test_net.py --dataset voc2007 --cfg configs/baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml \
         #--load_ckpt ./Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/May05-19-36-47_zdyhpc_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing

CUDA_VISIBLE_DEVICES=$cuda python tools/test_net.py --dataset voc2007 --cfg configs/baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml \
         --load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun12-04-43-48_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
         #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun09-09-26-11_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
         #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun12-08-53-08_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
         #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun12-04-45-59_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
         #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun10-20-45-32_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
         #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun09-09-26-11_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
         #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun09-09-13-42_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
         #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun07-02-46-04_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
         #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun06-01-44-15_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
         #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun06-01-44-15_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
         #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun07-02-46-04_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
#CUDA_VISIBLE_DEVICES=$cuda python tools/test_net.py --dataset voc2007 --cfg configs/baselines/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml \
        # --load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_faster_rcnn_X-101-64x4d-FPN_2x/Jun05-10-25-34_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
#CUDA_VISIBLE_DEVICES=$cuda python tools/test_net.py --dataset voc2007 --cfg configs/baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml \
        #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-101-64x4d-FPN_2x/Jun07-02-16-18_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
#        --load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-101-64x4d-FPN_2x/Jun06-01-41-46_zdy_step/ckpt/model_step$step.pth #--vis #--multi-gpu-testing
