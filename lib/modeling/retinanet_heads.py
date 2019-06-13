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

"""RetinaNet model heads and losses. See: https://arxiv.org/abs/1708.02002."""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import utils.net as net_utils
import math

from core.config import cfg


class fpn_retinanet_outputs(nn.Module):
    """Add RetinaNet on FPN specific outputs."""

    def __init__(self, dim_in, spatial_scales):
        super().__init__()
        self.dim_out = dim_in
        self.dim_in = dim_in
        self.spatial_scales = spatial_scales
        self.dim_out = self.dim_in
        self.num_anchors = len(cfg.RETINANET.ASPECT_RATIOS) * \
            cfg.RETINANET.SCALES_PER_OCTAVE

        # Create conv ops shared by all FPN levels
        self.n_conv_fpn_cls_modules = nn.ModuleList()
        self.n_conv_fpn_bbox_modules = nn.ModuleList()
        for nconv in range(cfg.RETINANET.NUM_CONVS):
            self.n_conv_fpn_cls_modules.append(
                nn.Conv2d(self.dim_in, self.dim_out, 3, 1, 1))
            self.n_conv_fpn_bbox_modules.append(
                nn.Conv2d(self.dim_in, self.dim_out, 3, 1, 1))

        cls_pred_dim = cfg.MODEL.NUM_CLASSES if cfg.RETINANET.SOFTMAX \
            else (cfg.MODEL.NUM_CLASSES - 1)

        # unpacked bbox feature and add prediction layers
        self.bbox_regr_dim = 4 * (cfg.MODEL.NUM_CLASSES - 1) \
            if cfg.RETINANET.CLASS_SPECIFIC_BBOX else 4

        self.fpn_cls_score = nn.Conv2d(self.dim_out,
                                       cls_pred_dim * self.num_anchors, 3, 1, 1)
        self.fpn_bbox_score = nn.Conv2d(self.dim_out,
                                        self.bbox_regr_dim * self.num_anchors, 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        for child_m in self.children():
            if isinstance(child_m, nn.ModuleList):
                child_m.apply(init_func)

        init.normal_(self.fpn_cls_score.weight, std=0.01)
        init.constant_(self.fpn_cls_score.bias,
                       -math.log((1 - cfg.RETINANET.PRIOR_PROB) / cfg.RETINANET.PRIOR_PROB))

        init.normal_(self.fpn_bbox_score.weight, std=0.01)
        init.constant_(self.fpn_bbox_score.bias, 0)

    def detectron_weight_mapping(self):
        k_min = cfg.FPN.RPN_MIN_LEVEL
        mapping_to_detectron = {
            'n_conv_fpn_cls_modules.0.weight': 'retnet_cls_conv_n0_fpn%d_w' % k_min,
            'n_conv_fpn_cls_modules.0.bias': 'retnet_cls_conv_n0_fpn%d_b' % k_min,
            'n_conv_fpn_cls_modules.1.weight': 'retnet_cls_conv_n1_fpn%d_w' % k_min,
            'n_conv_fpn_cls_modules.1.bias': 'retnet_cls_conv_n1_fpn%d_b' % k_min,
            'n_conv_fpn_cls_modules.2.weight': 'retnet_cls_conv_n2_fpn%d_w' % k_min,
            'n_conv_fpn_cls_modules.2.bias': 'retnet_cls_conv_n2_fpn%d_b' % k_min,
            'n_conv_fpn_cls_modules.3.weight': 'retnet_cls_conv_n3_fpn%d_w' % k_min,
            'n_conv_fpn_cls_modules.3.bias': 'retnet_cls_conv_n3_fpn%d_b' % k_min,

            'n_conv_fpn_bbox_modules.0.weight': 'retnet_bbox_conv_n0_fpn%d_w' % k_min,
            'n_conv_fpn_bbox_modules.0.bias': 'retnet_bbox_conv_n0_fpn%d_b' % k_min,
            'n_conv_fpn_bbox_modules.1.weight': 'retnet_bbox_conv_n1_fpn%d_w' % k_min,
            'n_conv_fpn_bbox_modules.1.bias': 'retnet_bbox_conv_n1_fpn%d_b' % k_min,
            'n_conv_fpn_bbox_modules.2.weight': 'retnet_bbox_conv_n2_fpn%d_w' % k_min,
            'n_conv_fpn_bbox_modules.2.bias': 'retnet_bbox_conv_n2_fpn%d_b' % k_min,
            'n_conv_fpn_bbox_modules.3.weight': 'retnet_bbox_conv_n3_fpn%d_w' % k_min,
            'n_conv_fpn_bbox_modules.3.bias': 'retnet_bbox_conv_n3_fpn%d_b' % k_min,

            'fpn_cls_score.weight': 'retnet_cls_pred_fpn%d_w' % k_min,
            'fpn_cls_score.bias': 'retnet_cls_pred_fpn%d_b' % k_min,
            'fpn_bbox_score.weight': 'retnet_bbox_pred_fpn%d_w' % k_min,
            'fpn_bbox_score.bias': 'retnet_bbox_pred_fpn%d_b' % k_min
        }
        return mapping_to_detectron, []

    def forward(self, blobs_in):
        k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
        k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
        assert len(blobs_in) == k_max - k_min + 1
        bbox_feat_list = []
        cls_score = []
        bbox_pred = []

        # ==========================================================================
        # classification tower with logits and prob prediction
        # ==========================================================================
        for lvl in range(k_min, k_max + 1):
            bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
            # classification tower stack convolution starts
            for nconv in range(cfg.RETINANET.NUM_CONVS):
                bl_out = self.n_conv_fpn_cls_modules[nconv](bl_in)
                bl_in = F.relu(bl_out, inplace=True)
                bl_feat = bl_in

            # cls tower stack convolution ends. Add the logits layer now
            retnet_cls_pred = self.fpn_cls_score(bl_feat)

            if not self.training:
                if cfg.RETINANET.SOFTMAX:
                    raise NotImplementedError("To be implemented")
                else:  # sigmoid
                    retnet_cls_probs = retnet_cls_pred.sigmoid()
                    cls_score.append(retnet_cls_probs)
            else:
                cls_score.append(retnet_cls_pred)

            if cfg.RETINANET.SHARE_CLS_BBOX_TOWER:
                bbox_feat_list.append(bl_feat)

        # ==========================================================================
        # bbox tower if not sharing features with the classification tower with
        # logits and prob prediction
        # ==========================================================================
        if not cfg.RETINANET.SHARE_CLS_BBOX_TOWER:
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                # classification tower stack convolution starts
                for nconv in range(cfg.RETINANET.NUM_CONVS):
                    bl_out = self.n_conv_fpn_bbox_modules[nconv](bl_in)
                    bl_in = F.relu(bl_out, inplace=True)
                    # Add octave scales and aspect ratio
                    # At least 1 convolution for dealing different aspect ratios
                    bl_feat = bl_in
                bbox_feat_list.append(bl_feat)

        # Depending on the features [shared/separate] for bbox, add prediction layer
        for i, lvl in enumerate(range(k_min, k_max + 1)):
            bl_feat = bbox_feat_list[i]
            retnet_bbox_pred = self.fpn_bbox_score(bl_feat)
            bbox_pred.append(retnet_bbox_pred)

        return cls_score, bbox_pred


def add_fpn_retinanet_losses(cls_score, bbox_pred, **kwargs):
    k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid

    losses_cls = []
    losses_bbox = []
    for i, lvl in enumerate(range(k_min, k_max + 1)):
        slvl = str(lvl)
        h, w = cls_score[i].shape[2:]
        retnet_cls_labels_fpn = kwargs['retnet_cls_labels_fpn' +
                                       slvl][:, :, :h, :w]
        retnet_bbox_targets_fpn = kwargs['retnet_roi_bbox_targets_fpn' +
                                         slvl][:, :, :, :h, :w]
        retnet_bbox_inside_weights_fpn = kwargs['retnet_bbox_inside_weights_wide_fpn' +
                                                slvl][:, :, :, :h, :w]
        retnet_fg_num = kwargs['retnet_fg_num']

        # ==========================================================================
        # bbox regression loss - SelectSmoothL1Loss for multiple anchors at a location
        # ==========================================================================
        bbox_loss = net_utils.select_smooth_l1_loss(
            bbox_pred[i], retnet_bbox_targets_fpn,
            retnet_bbox_inside_weights_fpn,
            retnet_fg_num,
            beta=cfg.RETINANET.BBOX_REG_BETA)

        # ==========================================================================
        # cls loss - depends on softmax/sigmoid outputs
        # ==========================================================================
        if cfg.RETINANET.SOFTMAX:
            raise NotImplementedError("To be implemented")
        else:
            cls_loss = net_utils.sigmoid_focal_loss(
                cls_score[i], retnet_cls_labels_fpn.float(),
                cfg.MODEL.NUM_CLASSES, retnet_fg_num, alpha=cfg.RETINANET.LOSS_ALPHA,
                gamma=cfg.RETINANET.LOSS_GAMMA
            )

        losses_bbox.append(bbox_loss)
        losses_cls.append(cls_loss)

    return losses_cls, losses_bbox
