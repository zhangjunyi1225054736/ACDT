#: Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import numpy as np
import pycocotools.mask as mask_util

from torch.autograd import Variable
import torch

from core.config import cfg
from utils.timer import Timer
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import utils.image as image_utils
import utils.keypoints as keypoint_utils
from numpy import random
import json
def im_detect_all(model, im,  box_proposals=None, timers=None, filename=None):


    """Process the outputs of model for testing
    Args:
      model: the network module
      im_data: Pytorch variable. Input batch to the model.
      im_info: Pytorch variable. Input batch to the model.
      gt_boxes: Pytorch variable. Input batch to the model.
      num_boxes: Pytorch variable. Input batch to the model.
      args: arguments from command line.
      timer: record the cost of time for different steps
    The rest of inputs are of type pytorch Variables and either input to or output from the model.
    """
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        scores, boxes, im_scale, blob_conv = im_detect_bbox_aug(
            model, im, box_proposals,filename)
    else:
        scores, boxes, im_scale, blob_conv = im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals)
    timers['im_detect_bbox'].toc()

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class) (numpy.ndarray)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    #print("boxes:",boxes)
    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            masks = im_detect_mask_aug(model, im, boxes, im_scale, blob_conv)
        else:
            masks = im_detect_mask(model, im_scale, boxes, blob_conv)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_keypoints'].tic()
        if cfg.TEST.KPS_AUG.ENABLED:
            heatmaps = im_detect_keypoints_aug(model, im, boxes, im_scale, blob_conv)
        else:
            heatmaps = im_detect_keypoints(model, im_scale, boxes, blob_conv)
        timers['im_detect_keypoints'].toc()

        timers['misc_keypoints'].tic()
        cls_keyps = keypoint_results(cls_boxes, heatmaps, boxes)
        timers['misc_keypoints'].toc()
    else:
        cls_keyps = None
    #print("cls_boxes shape:",cls_boxes)
    return cls_boxes, cls_segms, cls_keyps


def im_conv_body_only(model, im, target_scale, target_max_size):
    inputs, im_scale = _get_blobs(im, None, target_scale, target_max_size)

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = Variable(torch.from_numpy(inputs['data']), volatile=True).cuda()
    else:
        inputs['data'] = torch.from_numpy(inputs['data']).cuda()
    inputs.pop('im_info')

    blob_conv = model.module.convbody_net(**inputs)

    return blob_conv, im_scale


def im_detect_bbox(model, im, target_scale, target_max_size, boxes=None, region_box=None):
    """Prepare the bbox for testing"""
    h = im.shape[0]
    w = im.shape[1]

    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)
    #print("inputs['data']:",inputs['data'])
    #print("im_scale:", im_scale)
    #print("im_scale len:", len(im_scale))
    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
    else:
        inputs['data'] = [torch.from_numpy(inputs['data'])]
        inputs['im_info'] = [torch.from_numpy(inputs['im_info'])]

    return_dict = model(**inputs)
    #boxes_cp = boxes.copy()
    #rois = return_dict['rois'].data.cpu().numpy()
    #boxes_cp = rois[:, 1:5] / im_scale
    #edge_index = (boxes_cp[:,0]<2)|(boxes_cp[:,1]<2) | (boxes_cp[:,2] > w-3) | (boxes_cp[:,3] > h-3)
    #print("edge_index:",edge_index)
    scores = return_dict['cls_score'].data.cpu().numpy().squeeze()

    if cfg.MODEL.FASTER_RCNN:
        rois = return_dict['rois'].data.cpu().numpy()
        # unscale back to raw image space
        if region_box == None:
             #print("1111111111111")
             boxes = rois[:, 1:5] / im_scale

        else:
             #print(rois)
             boxes = rois[:, 1:5] / im_scale
             #print("boxes111:",boxes)
             #print("111:",boxes.shape) #[1000 * 4]
             edge_index = (boxes[:,0]<1)|(boxes[:,1]<1) | (boxes[:,2] > w-1) | (boxes[:,3] > h-1)
             #boxes = boxes[~edge_index]
             scores[edge_index] = [0 for i in range(13)]
             #print("222:",boxes.shape) #[1000 * 4]
             x_1 = region_box[0]
             y_1 = region_box[1]
             #print("x_1,y_1:",x_1,y_1)
             #boxes = np.reshape(boxes, (-1,4))
             boxes[:,(0,2)] = boxes[:,(0,2)] + x_1 - 1
             #print("333:",boxes)
             boxes[:,(1,3)] = boxes[:,(1,3)] + y_1 - 1
             #print("boxes222:",boxes)
    # cls prob (activations after softmax)
    # scores = return_dict['cls_score'].data.cpu().numpy().squeeze()
    #print("scores shape...:", scores.shape) # [1000 * 13]
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])
    #print("scores shape:", scores.shape)  # [1000 * 13]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = return_dict['bbox_pred'].data.cpu().numpy().squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # (legacy) Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
                         + cfg.TRAIN.BBOX_NORMALIZE_MEANS
        pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        #print("pred_boxes shape:",pred_boxes.shape) # [1000 * 52]
        #pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
    #print("pred_boxes shape:",pred_boxes.shape)
    #pred_boxes = pred_boxes[~edge_index]
    #print("pred_boxes shape...:",pred_boxes.shape)
    return scores, pred_boxes, im_scale, return_dict['blob_conv']


def im_detect_bbox_aug(model, im, box_proposals=None, filename = None):
    #print("im shape:",im.shape)
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'
    assert not cfg.MODEL.FASTER_RCNN or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Union heuristic must be used to combine Faster RCNN predictions'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _ = im_detect_bbox_hflip(
            model,
            im,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals
        )
        add_preds_t(scores_hf, boxes_hf)

    # Perform detection on croped images
    if cfg.TEST.BBOX_AUG.CROP:
    #if False:
       #print("crop image")
       h = im.shape[0]
       w = im.shape[1]
       if cfg.TEST.BBOX_AUG.RANDOM_CROP:
          #print("w,h:",w,h)
          random_crop_number = 4
          Difficult_regions = get_randomCropRegion(w,h,random_crop_number)
          #print("Difficult_regions:", Difficult_regions)
       elif cfg.TEST.BBOX_AUG.NOR_CROP:
          '''
          Difficult_regions = [[] for i in range(4)]
          Difficult_regions[0] = [1, 1, int(w/2), int(h/2)]
          Difficult_regions[1] = [int(w/2), 1, w-1, int(h/2)]
          Difficult_regions[2] = [1, int(h/2), int(w/2), h-1]
          Difficult_regions[3] = [int(w/2), int(h/2), w-1, h-1]
          '''
          w_n = 2
          h_n = 2

          region_n = w_n * h_n
          w_ = int(w / w_n) 
          h_ = int(h / h_n) 

          Difficult_regions = [[] for i in range(region_n)] 
          for i in range(region_n):
              j = int(i / w_n)
              k = i - j*w_n
              #print("j,k",j,k)
              Difficult_regions[i] =[k*w_+1, j*h_+1, (k+1)*w_, (j+1)*h_]
          #print(Difficult_regions) 
       elif cfg.TEST.BBOX_AUG.TRAINED_CROP:
          Difficult_regions = get_trainedCropRegion(filename)
          Difficult_regions = get_final_regions(w,h,Difficult_regions)
          Difficult_regions = Difficult_regions.tolist()

       i = 0
       max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
       f_crop_list = open("random_crop_list.txt",'a')
       temp_avg = []
       for region_box in Difficult_regions:
           #for j in range(4):
               #region_box[i] = int(region_box[i])
           #print("region box:", region_box)
           y1 = int(region_box[1]) 
           y2 = int(region_box[3]) 
           x1 = int(region_box[0])
           x2 = int(region_box[2])
           w = x2 - x1
           h = y2 - y1
           s_ = filename.split(".")[0]+'_'+str(i) + "," +str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," + str(w) + ","+ str(h) + "," + str(round(w/h,2))
           f_crop_list.write(s_+'\n')
           #if (w / h > 1.7) or (h / w  > 1.7):
              #continue 
           image_H = im.shape[0]
           #print("image_H:",image_H) 
           if y2 >= image_H or y1 >=image_H:
              continue
           im_crop = im[y1:y2, x1:x2] #crop the image
           #print("im_crop shape:", im_crop.shape)
           #if(region_box[1]<0):
           #cv2.imwrite("./crop/"+str(i)+'_'+filename, im_crop)
              # cv2.imwrite("./crop/or"+str(i)+".jpg", im)
           i = i + 1
           #print("im shape:",im.shape)
           #print("region_box:",region_box)
           #scale = min(im_crop.shape[0],im_crop.shape[1]) * 1
           if cfg.TEST.BBOX_AUG.TRAINED_CROP:
               #if im_crop.shape[0]/im_crop.shape[1]>4 or im_crop.shape[0]/im_crop.shape[1]<0.25:
               #    continue
               #r = round(random.uniform(1.3,2),2)
               #r = 1.5
               #scale = min(im_crop.shape[0],im_crop.shape[1]) * r
               #scores_scl, boxes_scl = im_detect_bbox_crop(model, im_crop, scale, max_size, box_proposals, region_box)
               scores_scl, boxes_scl = im_detect_bbox_crop(model, im_crop, cfg.TEST.SCALE, max_size, box_proposals, region_box)
           else:
               #scale = min(im_crop.shape[0],im_crop.shape[1]) * 1.5
               scores_scl, boxes_scl = im_detect_bbox_crop(model, im_crop, cfg.TEST.SCALE, max_size, box_proposals, region_box)
               #scores_scl, boxes_scl = im_detect_bbox_crop(model, im_crop, scale, max_size, box_proposals, region_box)
           #print("boxes_scl:",boxes_scl)
           #print("size scores_scl:", scores_scl.shape) #[1000, 13]
           #print("size boxes_scl:", boxes_scl.shape) #[1000, 52]
           #scores_scl = scores_scl + 0.05
           #score_result = get_scores(scores_scl) 
           #print("min score scl:", min(score_result))        
           #print("max score scl:", max(score_result))
           #temp_avg.append(np.mean(score_result))        
           #print("mean score scl:", np.mean(score_result))        
           #scores_scl, boxes_scl, cls_boxes = box_results_with_nms_and_limit(scores_scl, boxes_scl)
           #print("size scores_scl:", scores_scl.shape) #[1000, 13]
           #print("size boxes_scl:", boxes_scl.shape) #[1000, 52]
           #scores_scl = scores_scl + 0.1
           #index = nms_filter(scores_scl, boxes_scl)
           #print("index:", index)
           #print(" len index:", len(index))
           #add_preds_t(scores_scl[index,:], boxes_scl[index,:])
           add_preds_t(scores_scl, boxes_scl)

       #print("mean score scl:", np.mean(temp_avg))        
    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals
        )
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True
            )
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals
        )
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
                model, im, aspect_ratio, box_proposals, hflip=True
            )
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scale_i, blob_conv_i = im_detect_bbox(
        model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
    )
    #print("type scores_i:",type(scores_i))
    #print("size scores_i:",scores_i.shape)
    #print("min scores_i:",min(scores_i))
    #print("len scores_i:",len(scores_i))
    #score_result = get_scores(scores_i)
    #print("min score i:", min(score_result))
    #print("max score i:", max(score_result))
    #print("mean score i:", np.mean(score_result))

    add_preds_t(scores_i, boxes_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR)
        )

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
        )

    return scores_c, boxes_c, im_scale_i, blob_conv_i

def get_scores(scores):
    result = []
    for i in scores:
        if max(i) >=0.05:
           result.append(max(i))
        #if max(i) == 0:
        #print ("mean scores:", np.mean(i))
    return result
   
def get_randomCropRegion(width,height,crop_number=4):
    
    region_box = [[] for i in range(crop_number)]
    count = 0
    while 1:
        if count >= crop_number:
           break
        w = random.uniform(0.7 * width, width)
        h = random.uniform(0.7 * height, height)
        #w = width - 10
        #h = height - 10
        if h / w < 0.5 or h / w > 2:
             continue

        left = random.uniform(width - w)
        top = random.uniform(height - h)
        box = [int(left), int(top), int(left+w), int(top+h)]
        region_box[count] = box
        count = count + 1
    return region_box 

def get_trainedCropRegion(filename):
    with open('/data/zhangjunyi/drone-object-detection/prop.json', 'r') as f:
        data = json.load(f)
    bbox = data['bbox']
    name = data['img_name']
    region_box = bbox[name.index(filename)]
    return region_box


def get_final_regions(W, H, Difficult_regions):

    regions = Difficult_regions.copy()
    #print(regions)

    for i, bbox in enumerate(Difficult_regions):
        if len(bbox) != 4:
           print(bbox)
           continue
        x1,y1,x2,y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if (h / w) < 0.5:
           del regions[i]
           m_x = int((x1 + x2) / 2)
           box1 = [x1,y1,m_x,y2]
           box2 = [m_x,y1,x2,y2]
           regions.append(box1)
           regions.append(box2)
        elif (h / w) > 2: 
           del regions[i]
           m_y = int((y1 + y2) / 2)
           box1 = [x1,y1,x2,m_y]
           box2 = [x1,m_y,x2,y2]
           regions.append(box1)
           regions.append(box2)
        else:
           pass
    regions = np.array(regions)
    #print(regions)
    #center_x = (regions[:,0]+regions[:,2])/2
    #center_y = (regions[:,1]+regions[:,3])/2
    #center = np.hstack((np.array([center_x]).T,np.array([center_y]).T))
    #print("center:", center)
    w_ = 0.6*W
    h_ = 0.6*H

    for i, bbox in enumerate(regions):
        if len(bbox) != 4:
           print(bbox)
           np.delete(regions,i)
           continue

        x1,y1,x2,y2 = bbox
        w = x2 - x1
        h = y2 - y1

        #center_x,center_y = center[i]
        #center_x = int(center_x)
        #center_y = int(center_y)
        #min_y = min(center_y, Y-center_y) 
        #print("w,h",w,h)
        #print("w_,h_",w_,h_)
        if w >= w_ and h < h_:
           add_y = int((h_ - h) / 2)
           y1_new = y1 - add_y
           y2_new = y2 + add_y
           
           if y1_new < 0 :
              y1_new = 0
           if y2_new > H :
              y2_new = H-1

           regions[i][1] = y1_new
           regions[i][3] = y2_new
           

        elif w < w_ and h >= h_:
           add_x = int((w_ - w) / 2)
           x1_new = x1 - add_x
           x2_new = x2 + add_x
           
           if x1_new < 0 :
              x1_new = 0
           if x2_new > W:
              x2_new = W-1

           regions[i][0] = x1_new
           regions[i][2] = x2_new

        elif w < w_ and h < h_:
           #print("x1,y1,x2,y2:",x1,y1,x2,y2) 
           add_y = int((h_ - h) / 2)
           #print("add_y:",add_y)
           y1_new = y1 - add_y
           y2_new = y2 + add_y
           
           if y1_new < 0 :
              y1_new = 0
           if y2_new > H:
              y2_new = H-1
           
           add_x = int((w_ - w) / 2)
           #print("add_x:",add_x)
           x1_new = x1 - add_x
           x2_new = x2 + add_x
           
           if x1_new < 0 :
              x1_new = 0
           if x2_new > W:
              x2_new = W-1
           regions[i][1] = y1_new
           regions[i][3] = y2_new

           regions[i][0] = x1_new
           regions[i][2] = x2_new
        else:
           pass
        #print("regions [i]:", regions[i])
           #center_x,center_y = center[i]
           
    #print(len(regions)) 
    #print(regions)
    #exit()
    return regions


def im_detect_bbox_hflip(
        model, im, target_scale, target_max_size, box_proposals=None,region_box=None):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    im_hf = im[:, ::-1, :]
    im_width = im.shape[1]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)
    else:
        box_proposals_hf = None

    scores_hf, boxes_hf, im_scale, _ = im_detect_bbox(
        model, im_hf, target_scale, target_max_size, boxes=box_proposals_hf, region_box=region_box
    )

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)

    return scores_hf, boxes_inv, im_scale


def im_detect_bbox_scale(
        model, im, target_scale, target_max_size, box_proposals=None, hflip=False):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, target_scale, target_max_size, box_proposals=box_proposals
        )
    else:
        scores_scl, boxes_scl, _, _ = im_detect_bbox(
            model, im, target_scale, target_max_size, boxes=box_proposals
        )
    return scores_scl, boxes_scl

def im_detect_bbox_crop(model, im, target_scale, target_max_size, box_proposals=None, region_box=None, hflip=False):
    """
    Computes bbox detections at the crop image
    """
    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, target_scale, target_max_size, box_proposals=box_proposalsi, region_box=region_box
        )
    else:
        scores_scl, boxes_scl, _, _ = im_detect_bbox(
            model, im, target_scale, target_max_size, boxes=box_proposals, region_box=region_box
        )
    return scores_scl, boxes_scl
   

def im_detect_bbox_aspect_ratio(
        model, im, aspect_ratio, box_proposals=None, hflip=False):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)
    else:
        box_proposals_ar = None

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals_ar
        )
    else:
        scores_ar, boxes_ar, _, _ = im_detect_bbox(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            boxes=box_proposals_ar
        )

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


def im_detect_mask(model, im_scale, boxes, blob_conv):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scale (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)
        blob_conv (Variable): base features from the backbone network.

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    pred_masks = model.module.mask_net(blob_conv, inputs)
    pred_masks = pred_masks.data.cpu().numpy().squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
    else:
        pred_masks = pred_masks.reshape([-1, 1, M, M])

    return pred_masks


def im_detect_mask_aug(model, im, boxes, im_scale, blob_conv):
    """Performs mask detection with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes
        im_scale (list): image blob scales as returned by im_detect_bbox
        blob_conv (Tensor): base features from the backbone network.

    Returns:
        masks (ndarray): R x K x M x M array of class specific soft masks
    """
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    masks_ts = []

    # Compute masks for the original image (identity transform)
    masks_i = im_detect_mask(model, im_scale, boxes, blob_conv)
    masks_ts.append(masks_i)

    # Perform mask detection on the horizontally flipped image
    if cfg.TEST.MASK_AUG.H_FLIP:
        masks_hf = im_detect_mask_hflip(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes
        )
        masks_ts.append(masks_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
        max_size = cfg.TEST.MASK_AUG.MAX_SIZE
        masks_scl = im_detect_mask_scale(model, im, scale, max_size, boxes)
        masks_ts.append(masks_scl)

        if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
            masks_scl_hf = im_detect_mask_scale(
                model, im, scale, max_size, boxes, hflip=True
            )
            masks_ts.append(masks_scl_hf)

    # Compute masks at different aspect ratios
    for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
        masks_ar = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes)
        masks_ts.append(masks_ar)

        if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
            masks_ar_hf = im_detect_mask_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            masks_ts.append(masks_ar_hf)

    # Combine the predicted soft masks
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        masks_c = np.mean(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        masks_c = np.amax(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':

        def logit(y):
            return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))

        logit_masks = [logit(y) for y in masks_ts]
        logit_masks = np.mean(logit_masks, axis=0)
        masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR)
        )

    return masks_c


def im_detect_mask_hflip(model, im, target_scale, target_max_size, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    blob_conv, im_scale = im_conv_body_only(model, im_hf, target_scale, target_max_size)
    masks_hf = im_detect_mask(model, im_scale, boxes_hf, blob_conv)

    # Invert the predicted soft masks
    masks_inv = masks_hf[:, :, :, ::-1]

    return masks_inv


def im_detect_mask_scale(
        model, im, target_scale, target_max_size, boxes, hflip=False):
    """Computes masks at the given scale."""
    if hflip:
        masks_scl = im_detect_mask_hflip(
            model, im, target_scale, target_max_size, boxes
        )
    else:
        blob_conv, im_scale = im_conv_body_only(model, im, target_scale, target_max_size)
        masks_scl = im_detect_mask(model, im_scale, boxes, blob_conv)
    return masks_scl


def im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio."""

    # Perform mask detection on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        masks_ar = im_detect_mask_hflip(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes_ar
        )
    else:
        blob_conv, im_scale = im_conv_body_only(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
        )
        masks_ar = im_detect_mask(model, im_scale, boxes_ar, blob_conv)

    return masks_ar


def im_detect_keypoints(model, im_scale, boxes, blob_conv):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scale (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    M = cfg.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    pred_heatmaps = model.module.keypoint_net(blob_conv, inputs)
    pred_heatmaps = pred_heatmaps.data.cpu().numpy().squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps


def im_detect_keypoints_aug(model, im, boxes, im_scale, blob_conv):
    """Computes keypoint predictions with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes
        im_scale (list): image blob scales as returned by im_detect_bbox
        blob_conv (Tensor): base features from the backbone network.

    Returns:
        heatmaps (ndarray): R x J x M x M array of keypoint location logits
    """
    # Collect heatmaps predicted under different transformations
    heatmaps_ts = []
    # Tag predictions computed under downscaling and upscaling transformations
    ds_ts = []
    us_ts = []

    def add_heatmaps_t(heatmaps_t, ds_t=False, us_t=False):
        heatmaps_ts.append(heatmaps_t)
        ds_ts.append(ds_t)
        us_ts.append(us_t)

    # Compute the heatmaps for the original image (identity transform)
    heatmaps_i = im_detect_keypoints(model, im_scale, boxes, blob_conv)
    add_heatmaps_t(heatmaps_i)

    # Perform keypoints detection on the horizontally flipped image
    if cfg.TEST.KPS_AUG.H_FLIP:
        heatmaps_hf = im_detect_keypoints_hflip(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.KPS_AUG.SCALES:
        ds_scl = scale < cfg.TEST.SCALE
        us_scl = scale > cfg.TEST.SCALE
        heatmaps_scl = im_detect_keypoints_scale(
            model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_scl, ds_scl, us_scl)

        if cfg.TEST.KPS_AUG.SCALE_H_FLIP:
            heatmaps_scl_hf = im_detect_keypoints_scale(
                model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_scl_hf, ds_scl, us_scl)

    # Compute keypoints at different aspect ratios
    for aspect_ratio in cfg.TEST.KPS_AUG.ASPECT_RATIOS:
        heatmaps_ar = im_detect_keypoints_aspect_ratio(
            model, im, aspect_ratio, boxes
        )
        add_heatmaps_t(heatmaps_ar)

        if cfg.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP:
            heatmaps_ar_hf = im_detect_keypoints_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_ar_hf)

    # Select the heuristic function for combining the heatmaps
    if cfg.TEST.KPS_AUG.HEUR == 'HM_AVG':
        np_f = np.mean
    elif cfg.TEST.KPS_AUG.HEUR == 'HM_MAX':
        np_f = np.amax
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.KPS_AUG.HEUR)
        )

    def heur_f(hms_ts):
        return np_f(hms_ts, axis=0)

    # Combine the heatmaps
    if cfg.TEST.KPS_AUG.SCALE_SIZE_DEP:
        heatmaps_c = combine_heatmaps_size_dep(
            heatmaps_ts, ds_ts, us_ts, boxes, heur_f
        )
    else:
        heatmaps_c = heur_f(heatmaps_ts)

    return heatmaps_c


def im_detect_keypoints_hflip(model, im, target_scale, target_max_size, boxes):
    """Computes keypoint predictions on the horizontally flipped image.
    Function signature is the same as for im_detect_keypoints_aug.
    """
    # Compute keypoints for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    blob_conv, im_scale = im_conv_body_only(model, im_hf, target_scale, target_max_size)
    heatmaps_hf = im_detect_keypoints(model, im_scale, boxes_hf, blob_conv)

    # Invert the predicted keypoints
    heatmaps_inv = keypoint_utils.flip_heatmaps(heatmaps_hf)

    return heatmaps_inv


def im_detect_keypoints_scale(
    model, im, target_scale, target_max_size, boxes, hflip=False):
    """Computes keypoint predictions at the given scale."""
    if hflip:
        heatmaps_scl = im_detect_keypoints_hflip(
            model, im, target_scale, target_max_size, boxes
        )
    else:
        blob_conv, im_scale = im_conv_body_only(model, im, target_scale, target_max_size)
        heatmaps_scl = im_detect_keypoints(model, im_scale, boxes, blob_conv)
    return heatmaps_scl


def im_detect_keypoints_aspect_ratio(
    model, im, aspect_ratio, boxes, hflip=False):
    """Detects keypoints at the given width-relative aspect ratio."""

    # Perform keypoint detectionon the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        heatmaps_ar = im_detect_keypoints_hflip(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes_ar
        )
    else:
        blob_conv, im_scale = im_conv_body_only(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
        )
        heatmaps_ar = im_detect_keypoints(model, im_scale, boxes_ar, blob_conv)

    return heatmaps_ar


def combine_heatmaps_size_dep(hms_ts, ds_ts, us_ts, boxes, heur_f):
    """Combines heatmaps while taking object sizes into account."""
    assert len(hms_ts) == len(ds_ts) and len(ds_ts) == len(us_ts), \
        'All sets of hms must be tagged with downscaling and upscaling flags'

    # Classify objects into small+medium and large based on their box areas
    areas = box_utils.boxes_area(boxes)
    sm_objs = areas < cfg.TEST.KPS_AUG.AREA_TH
    l_objs = areas >= cfg.TEST.KPS_AUG.AREA_TH

    # Combine heatmaps computed under different transformations for each object
    hms_c = np.zeros_like(hms_ts[0])

    for i in range(hms_c.shape[0]):
        hms_to_combine = []
        for hms_t, ds_t, us_t in zip(hms_ts, ds_ts, us_ts):
            # Discard downscaling predictions for small and medium objects
            if sm_objs[i] and ds_t:
                continue
            # Discard upscaling predictions for large objects
            if l_objs[i] and us_t:
                continue
            hms_to_combine.append(hms_t[i])
        hms_c[i] = heur_f(hms_to_combine)

    return hms_c


def nms_filter(scores, boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)] 
    index = []
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        keep = box_utils.nms(dets_j, 0.7)
        index.extend(keep)
    return index
    

def box_results_with_nms_and_limit(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        #print("dets_j:", dets_j.shape)
        #keep = box_utils.nms(dets_j, cfg.TEST.NMS)
        #print("keep:",len(keep))
        #print("keep:",keep)
        #exit()
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            #print("keep:",keep)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    #print("score shape:",score.shape)
    #print("boxes shape:",boxes.shape)
    return scores, boxes, cls_boxes


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = (ref_box[2] - ref_box[0] + 1)
            h = (ref_box[3] - ref_box[1] + 1)
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            # For dumping to json, need to decode the byte string.
            # https://github.com/cocodataset/cocoapi/issues/70
            rle['counts'] = rle['counts'].decode('ascii')
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms


def keypoint_results(cls_boxes, pred_heatmaps, ref_boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()
    xy_preds = keypoint_utils.heatmaps_to_keypoints(pred_heatmaps, ref_boxes)

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        keep = keypoint_utils.nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps


def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_utils.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn_utils.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    #print("im shape:",im.shape)
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
            blob_utils.get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale
                          
