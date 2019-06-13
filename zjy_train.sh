rm -rf /data/zhangjunyi/drone-object-detection/data/cache

CUDA_VISIBLE_DEVICES=4,5,6,7 python  tools/train_net_step.py --dataset voc2007 --cfg configs/baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml  --bs 4 #--load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x/Jun09-09-26-11_zdy_step/ckpt/model_step54999.pth  #--lr 0.00001

#CUDA_VISIBLE_DEVICES=0,1,2,3 python  tools/train_net_step.py --dataset voc2007 --cfg configs/baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml  --bs 4 --load_ckpt /data/zhangjunyi/drone-object-detection/Outputs/voc2007/e2e_mask_rcnn_X-101-64x4d-FPN_2x/Jun06-01-41-46_zdy_step/ckpt/model_step64999.pth  #--lr 0.00001

