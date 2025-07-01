# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.layers import ShapeSpec
from .build import META_ARCH_REGISTRY
from ..backbone import build_backbone
from ..postprocessing import detector_postprocess, fullize_mask
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss
from detr.backbone import Joiner
from detr.detr import DETR, SetCriterion, GDETR
from detr.matcher import HungarianMatcher
from detr.position_encoding import PositionEmbeddingSine
from detr.transformer import Transformer, TransformerDecoderLayer, TransformerDecoder
from detr.segmentation import DETRsegm, GDETRsegm, PostProcessPanoptic, PostProcessSegm
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from util.misc import NestedTensor
from ..relation_heads import build_relation_heads
from detectron2.data.datasets.coco import convert_coco_poly_to_mask
from detectron2.utils.torch_utils import union_box_matrix,union_box,union_box_list,union_boxes
__all__ = ["Detr"]


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class Detr(nn.Module):
    """
    Implement Detr
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.detr = DETR(
            backbone, transformer, num_classes=self.num_classes, num_queries=num_queries, aux_loss=deep_supervision
        )
        if self.mask_on:
            frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
            if frozen_weights != '':
                print("LOAD pre-trained weights")
                weight = torch.load(frozen_weights, map_location=lambda storage, loc: storage)['model']
                new_weight = {}
                for k, v in weight.items():
                    if 'detr.' in k:
                        new_weight[k.replace('detr.', '')] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.detr.load_state_dict(new_weight)
                del new_weight
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ''))
            self.seg_postprocess = PostProcessSegm

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        if self.mask_on:
            losses += ["masks"]
        self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.instance_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.to(self.device)

    def forward(self, batched_inputs, iteration, mode, training, print_size=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        losses = {}
        results=[]
        metrics={}

        images = self.preprocess_image(batched_inputs)
        output, features, hs, src_proj, mask, pos  = self.detr(images)

        if training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            box_feature = output['pred_features']
            instance_detections,keeps = self.inference(box_cls, box_pred, mask_pred, box_feature, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(instance_detections, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            for i, r in enumerate(processed_results):
                if i >= len(results):
                    results.append({})
                results[i].update(processed_results[i])
            return results, losses, metrics

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({'masks': gt_masks})
        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, box_feature, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        keeps=[]
        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, box_feature_per_image, image_size) in enumerate(
                zip(
                        scores, labels, box_pred, box_feature, image_sizes
                )):
            keep = scores_per_image > self.instance_threshold
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.pred_features = box_feature_per_image
            result = result[keep]
            results.append(result)
            keeps.append(keep)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

@META_ARCH_REGISTRY.register()
class DetrRelationByInstanceCOSSimilarity(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.relation_classes = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        self.similarity=cfg.SIMILARITY
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        self.num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        self.detr = DETR(
            backbone, transformer, num_classes=self.num_classes, num_queries=self.num_queries, aux_loss=deep_supervision
        )
        if self.mask_on:
            frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
            if frozen_weights != '':
                print("LOAD pre-trained weights")
                weight = torch.load(frozen_weights, map_location=lambda storage, loc: storage)['model']
                new_weight = {}
                for k, v in weight.items():
                    if 'detr.' in k:
                        new_weight[k.replace('detr.', '')] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.detr.load_state_dict(new_weight)
                del new_weight
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ''))
            self.seg_postprocess = PostProcessSegm

        # building criterion
        matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        if self.mask_on:
            losses += ["masks"]
        self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.instance_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

        self.instance_encoder = None
        if cfg.MODEL.ROI_HEADS.ENCODE_DIM>0:
            self.instance_encoder = nn.Linear(hidden_dim,512)
        self.location_dim=2 #cfg.MODEL.ROI_HEADS.LOCATION_DIM
        self.location_encode = nn.Sequential(nn.Linear(self.location_dim, 256),nn.Linear(256, 512))

        self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST
        self.relation_heads = build_relation_heads(cfg)
        # self.positive_crowd_threshold = cfg.MODEL.ROI_HEADS.POSITIVE_CROWD_THRESHOLD
        # self.positive_box_threshold = cfg.MODEL.ROI_HEADS.POSITIVE_BOX_THRESHOLD

        self.gt_similarity_mode = cfg.MODEL.ROI_HEADS.GT_SIMILARITY_MODE
        self.similarity_mode = cfg.MODEL.ROI_HEADS.SIMILARITY_MODE
        self.graph_search_mode = cfg.MODEL.ROI_HEADS.GRAPH_SEARCH_MODE
        #
        # self.similarity_loss = cfg.MODEL.ROI_HEADS.SIMILARITY_LOSS
        # self.train_only_same_class = cfg.MODEL.ROI_HEADS.SIMILARITY_TRAIN_ONLY_SAME_CLASS

        self.to(self.device)

    def forward(self, batched_inputs,iteration, mode, training, print_size=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        self.print_size=print_size
        losses = {}
        results=[]
        metrics={}
        relation_results=None
        images = self.preprocess_image(batched_inputs)

        output, features, pos = self.detr(images)
        src, mask = features[-1].decompose()
        bs, c, h, w = src.shape

        if mode=="instance":
            if not training:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else None
                box_feature = output["pred_features"]
                instance_detections,keeps = self.inference(box_cls, box_pred, mask_pred, box_feature, images.image_sizes)
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes,
                                                     "instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(instance_results[i])
            else:
                print("not support instance training")
                exit()
        elif mode=="relation":
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            instance_local_features = output["pred_features"][-1]
            instance_detections, keeps = self.inference(box_cls, box_pred, mask_pred, instance_local_features,
                                                        images.image_sizes, self.instance_threshold)
            # self.fullize_mask(instance_detections)
            mask_pred = F.sigmoid(mask_pred)

            instance_location_features = []
            for pred_instance in instance_detections:
                if self.location_dim == 4:
                    global_location = torch.Tensor([0, 0, pred_instance.image_size[1], pred_instance.image_size[0]]).to(self.device)
                    global_location = global_location.unsqueeze(0).repeat(len(pred_instance), 1)
                    instance_location_feature = pred_instance.pred_boxes.tensor - global_location  # x1,y1,x2,y2
                    global_location = torch.Tensor([pred_instance.image_size[1], pred_instance.image_size[0], pred_instance.image_size[1],pred_instance.image_size[0]]).to(self.device)
                    global_location = global_location.unsqueeze(0).repeat(len(pred_instance), 1)
                    instance_location_feature = instance_location_feature / global_location
                elif self.location_dim == 2:
                    instance_location_feature = pred_instance.pred_boxes.tensor[:,2:] - pred_instance.pred_boxes.tensor[:, :2]  # x1,y1,x2,y2
                    global_location = torch.Tensor([pred_instance.image_size[1], pred_instance.image_size[0]]).to(self.device)
                    global_location = global_location.unsqueeze(0).repeat(len(pred_instance), 1)
                    instance_location_feature = instance_location_feature / global_location
                instance_location_features.append(self.location_encode(instance_location_feature))

            instance_local_features = [self.instance_encoder(box_feature[keep]) for box_feature, keep in zip(instance_local_features, keeps)]

            if training:
                gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
                gt_phrases = [x["phrases"].to(self.device) for x in batched_inputs]

                g_mask_gts = []
                for mask_pred_single, gt_crowd in zip(mask_pred, gt_crowds):
                    pred_mask = mask_pred_single
                    gt_mask = gt_crowd.gt_masks.tensor
                    gt_mask_reshape = F.interpolate(gt_mask.unsqueeze(0).float(), size=pred_mask.shape[1:], mode='bilinear', align_corners=False)
                    g_mask_gts.append(gt_mask_reshape[0])

                # instance_local_features_pad0=[]
                # instance_similarities = []
                # for pred_instance, instance_local_feature, instance_location_feature in zip(instance_detections,
                #                                                                             instance_local_features,
                #                                                                             instance_location_features):
                #     instance_feature = torch.cat([instance_local_feature, instance_location_feature], dim=1)
                #     class_similarity_score = (pred_instance.pred_classes.unsqueeze(1).repeat(1, len(pred_instance)) == pred_instance.pred_classes.unsqueeze(0).repeat(len(pred_instance),1)).float()
                #     # prime,attention=self.gat(instance_feature,class_similarity_score)
                #     simi_flatten = F.cosine_similarity(
                #         instance_feature.unsqueeze(1).repeat(1, instance_feature.shape[0], 1).view(instance_feature.shape[0] * instance_feature.shape[0], instance_feature.shape[1]),
                #         instance_feature.unsqueeze(0).repeat(instance_feature.shape[0], 1, 1).view(instance_feature.shape[0] * instance_feature.shape[0], instance_feature.shape[1]))
                #     instance_similarities.append((simi_flatten.view(instance_feature.shape[0], instance_feature.shape[0]) + 1) / 2)
                    # if self.similarity_mode=="att":
                    #     instance_similarities.append(attention)
                    # elif self.similarity_mode=="distance":
                    #     euclidean_distance = self.euclidean_dist(prime, prime)
                    #     instance_similarities.append(torch.exp(-euclidean_distance))
                    # elif self.similarity_mode=="cosine":
                    #     euclidean_distance = F.cosine_similarity(prime.unsqueeze(1).repeat(1,prime.shape[0],1).view(prime.shape[0]*prime.shape[0],prime.shape[1]),
                    #                                              prime.unsqueeze(0).repeat(prime.shape[0],1,1).view(prime.shape[0]*prime.shape[0],prime.shape[1]))
                    #     euclidean_distance=euclidean_distance.view(prime.shape[0],prime.shape[0])
                    #     instance_similarities.append((euclidean_distance+1)/2)
                    # else:
                    #     print("no such similarity function: " + self.similarity_mode)
                    #     exit()

                # if training:
                #     gt_instance_similarities = self.gt_similarity(instance_detections, gt_instances, gt_crowds)
                #     if self.train_only_same_class:
                #         gts, preds = [], []
                #         for pred_instance, gt_instance_similarity, instance_similarity in zip(instance_detections,
                #                                                                               gt_instance_similarities,
                #                                                                               instance_similarities):
                #             label_marix = pred_instance.pred_classes.unsqueeze(1).repeat(1, len(pred_instance))
                #             label_marix_1 = pred_instance.pred_classes.unsqueeze(0).repeat(len(pred_instance), 1)
                #             choose_index = (label_marix == label_marix_1)
                #             gts.append(gt_instance_similarity[choose_index])
                #             preds.append(instance_similarity[choose_index])
                #
                #         gts = torch.cat(gts)
                #         preds = torch.cat(preds)
                #     else:
                #         gts = torch.cat(gt_instance_similarities)
                #         preds = torch.cat(instance_similarities)
                #     print(class_similarity_score)
                #     print(gts)
                #     print(preds)
                #     print('------------------------------------------------------')
                #
                #     if self.similarity_loss == "f1":
                #         losses['loss_similarity'] = F.l1_loss(preds, gts)
                #     elif self.similarity_loss == "smooth_f1":
                #         losses['loss_similarity'] = F.smooth_l1_loss(preds, gts)
                #     elif self.similarity_loss == "bi_focal":
                #         losses['loss_pos_similarity'], losses['loss_neg_similarity'] = self.binary_focal_loss(preds,gts)

                # if self.print_size:
                #     if gt_instances:
                #         print("pred instances", len(instance_detections[0]), "gt instances", len(gt_instances[0]))
                #     else:
                #         print("pred instances", len(instance_detections[0]))

                if len(instance_detections[0]) > 0:
                    to_pred_phrases = self.get_target_from_instance_pred(mask_pred, keeps, instance_detections,g_mask_gts, gt_phrases)

                    if len(to_pred_phrases[0]) > 0:
                        relation_results, rela_losses, rela_metrics = self.relation_heads(instance_detections,
                                                                                          instance_local_features,
                                                                                          to_pred_phrases,
                                                                                          training=training,
                                                                                          iteration=iteration)
                        losses.update(rela_losses)
                    else:
                        losses = {}
                else:
                    losses = {}
            else:
                # instance_local_features_pad0=[]
                instance_similarities = []
                for pred_instance, instance_local_feature, instance_location_feature in zip(instance_detections,
                                                                                            instance_local_features,
                                                                                            instance_location_features):
                    instance_feature = torch.cat([instance_local_feature, instance_location_feature], dim=1)
                    class_similarity_score = (pred_instance.pred_classes.unsqueeze(1).repeat(1, len(pred_instance)) == pred_instance.pred_classes.unsqueeze(0).repeat(len(pred_instance),1)).float()
                    # prime,attention=self.gat(instance_feature,class_similarity_score)
                    simi_flatten = F.cosine_similarity(
                        instance_feature.unsqueeze(1).repeat(1, instance_feature.shape[0], 1).view(instance_feature.shape[0] * instance_feature.shape[0], instance_feature.shape[1]),
                        instance_feature.unsqueeze(0).repeat(instance_feature.shape[0], 1, 1).view(instance_feature.shape[0] * instance_feature.shape[0], instance_feature.shape[1]))
                    instance_similarities.append((simi_flatten.view(instance_feature.shape[0], instance_feature.shape[0]) + 1) / 2)
                    # print(class_similarity_score)
                    # print(instance_similarities)
                if len(instance_detections[0]) > 0:
                    to_pred_crowds = self.get_to_pred_crowds_from_instance_pred_with_similarity(mask_pred, keeps, instance_detections, instance_similarities)
                    to_pred_crowds, to_pred_phrases = self.generate_to_pred_phrases_from_to_pred_crowds(instance_detections, to_pred_crowds)
                    if len(to_pred_phrases[0]) > 0:
                        relation_results, rela_losses, rela_metrics = self.relation_heads(instance_detections,
                                                                                          instance_local_features,
                                                                                          to_pred_phrases,
                                                                                          training=training,
                                                                                          iteration=iteration)
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes,"instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(instance_results[i])
                    if relation_results:
                        results[i]['phrases']=relation_results[i]

        return results, losses, metrics

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks.tensor
                # gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({'masks': gt_masks})
        return new_targets

    def fullize_mask(self, instances):
        for instance_per_image in instances:
            height, width = instance_per_image.image_size
            fullize_mask(instance_per_image, height, width)

    def get_target_from_instance_pred(self,mask_preds, keeps, instance_detections, g_mask_gts, gt_phrases):
        to_pred_phrases=[]
        for mask_pred, keep, instance_detection, g_mask_gt, gt_phrase in zip(mask_preds, keeps, instance_detections, g_mask_gts, gt_phrases):
            pred_instance_boxes_pad0 = torch.cat([instance_detection.pred_boxes.tensor, torch.Tensor([[instance_detection.image_size[1], instance_detection.image_size[0], 0, 0]]).to(self.device)]).unsqueeze(0)
            to_pred_phrase=Instances(image_size=instance_detection.image_size)
            mask_pred=mask_pred[keep]>0.5
            g_mask_gt=g_mask_gt>0.5
            # phrase, instance
            instance_area = torch.sum(mask_pred.unsqueeze(0).repeat(len(gt_phrase), 1, 1, 1), dim=[2, 3])
            subject_interaction_area=torch.sum(mask_pred.unsqueeze(0).repeat(len(gt_phrase), 1, 1, 1)
                                               * g_mask_gt[gt_phrase.gt_subject_crowd_ids].unsqueeze(1).repeat(1, len(instance_detection), 1, 1), dim=[2, 3])
            gt_phrase_subject_instance_match_score = subject_interaction_area/instance_area \
                                                     * (instance_detection.pred_classes.unsqueeze(0).repeat(len(gt_phrase), 1)
                                                        ==gt_phrase.gt_subject_classes.unsqueeze(1).repeat(1, len(instance_detection)))
            object_interaction_area = torch.sum(mask_pred.unsqueeze(0).repeat(len(gt_phrase), 1, 1, 1) \
                                                * g_mask_gt[gt_phrase.gt_object_crowd_ids].unsqueeze(1).repeat(1, len(instance_detection), 1, 1), dim=[2, 3])
            gt_phrase_object_instance_match_score = object_interaction_area / instance_area \
                                             * (instance_detection.pred_classes.unsqueeze(0).repeat(len(gt_phrase), 1)
                                                == gt_phrase.gt_object_classes.unsqueeze(1).repeat(1, len(instance_detection)))
            subject_dict=defaultdict(list)
            object_dict = defaultdict(list)
            all_classes=torch.unique(instance_detection.pred_classes)
            max_len=0
            for cls in all_classes:
                cls_length=torch.sum(instance_detection.pred_classes==cls).item()
                max_len=max(cls_length,max_len)
            subject_classes = gt_phrase.gt_subject_classes
            object_classes = gt_phrase.gt_object_classes
            to_pred_phrase_pred_subject_lens = []
            to_pred_phrase_pred_object_lens = []
            to_pred_phrase_pred_boxes = []
            to_pred_phrase_pred_subject_classes = []
            to_pred_phrase_pred_object_classes = []
            to_pred_phrase_pred_subject_ids_list=[]
            to_pred_phrase_pred_object_ids_list = []
            to_pred_phrase_pred_subject_in_clust=[]
            to_pred_phrase_pred_object_in_clust = []
            to_pred_phrase_pred_subject_scores_list=[]
            to_pred_phrase_pred_object_scores_list = []
            to_pred_phrase_pred_relation_onehots_list=[]
            to_pred_phrase_confidence=[]
            for i,(single_gt_phrase_subject_instance_match_score,single_gt_phrase_object_instance_match_score) in enumerate(zip(gt_phrase_subject_instance_match_score,gt_phrase_object_instance_match_score)):
                to_pred_phrase_pred_subject_ids = torch.where(single_gt_phrase_subject_instance_match_score>0)[0]
                to_pred_phrase_pred_object_ids = torch.where(single_gt_phrase_object_instance_match_score > 0)[0]
                to_pred_phrase_pred_subject_len = torch.sum(to_pred_phrase_pred_subject_ids>=0)
                to_pred_phrase_pred_object_len = torch.sum(to_pred_phrase_pred_object_ids>=0)
                to_pred_phrase_pred_subject_scores = single_gt_phrase_subject_instance_match_score[to_pred_phrase_pred_subject_ids]
                to_pred_phrase_pred_object_scores = single_gt_phrase_object_instance_match_score[to_pred_phrase_pred_object_ids]
                if to_pred_phrase_pred_subject_len.item()<=0 or to_pred_phrase_pred_object_len.item()<=0:
                    continue
                to_pred_phrase_pred_subject_crowd_box=union_box_matrix(pred_instance_boxes_pad0[:,to_pred_phrase_pred_subject_ids])
                to_pred_phrase_pred_object_crowd_box = union_box_matrix(pred_instance_boxes_pad0[:,to_pred_phrase_pred_object_ids])
                to_pred_phrase_pred_box=union_boxes(to_pred_phrase_pred_subject_crowd_box,
                                                        to_pred_phrase_pred_object_crowd_box)[0]
                to_pred_phrase_pred_subject_classes.append(gt_phrase.gt_subject_classes[i])
                to_pred_phrase_pred_object_classes.append(gt_phrase.gt_object_classes[i])
                to_pred_phrase_pred_subject_ids_list.append(torch.cat([to_pred_phrase_pred_subject_ids,-torch.ones(max_len-len(to_pred_phrase_pred_subject_ids)).to(self.device)]))
                to_pred_phrase_pred_object_ids_list.append(torch.cat([to_pred_phrase_pred_object_ids,-torch.ones(max_len-len(to_pred_phrase_pred_object_ids)).to(self.device)]))
                to_pred_phrase_pred_subject_in_clust.append(to_pred_phrase_pred_subject_ids_list[-1]>=0)
                to_pred_phrase_pred_object_in_clust.append(to_pred_phrase_pred_object_ids_list[-1] >= 0)
                to_pred_phrase_pred_subject_lens.append(to_pred_phrase_pred_subject_len)
                to_pred_phrase_pred_object_lens.append(to_pred_phrase_pred_object_len)
                to_pred_phrase_pred_subject_scores_list.append(torch.cat([to_pred_phrase_pred_subject_scores,torch.zeros(max_len-len(to_pred_phrase_pred_subject_ids)).to(self.device)]))
                to_pred_phrase_pred_object_scores_list.append(torch.cat([to_pred_phrase_pred_object_scores,torch.zeros(max_len-len(to_pred_phrase_pred_object_ids)).to(self.device)]))
                to_pred_phrase_pred_boxes.append(to_pred_phrase_pred_box)
                to_pred_phrase_pred_relation_onehots_list.append(gt_phrase.gt_relation_onehots[i])
                to_pred_phrase_confidence.append(1-gt_phrase.gt_relation_onehots[i][0])
                subject_dict[subject_classes[i]].append(object_classes[i])
                object_dict[object_classes[i]].append(subject_classes[i])

            if len(to_pred_phrase_pred_subject_classes)>0:
                negative_sub_classes=[]
                negative_obj_classes = []
                for sub_cls in subject_dict:
                    exist_classes=torch.stack(subject_dict[sub_cls])
                    not_exist_classes=all_classes[torch.where(torch.sum(all_classes.unsqueeze(1).repeat(1,len(exist_classes))==exist_classes.unsqueeze(0).repeat(len(all_classes),1),dim=1)==0)]
                    for obj_cls in not_exist_classes:
                        negative_sub_classes.append(sub_cls)
                        negative_obj_classes.append(obj_cls)

                exist_sub_classes = torch.Tensor([]).to(self.device)
                if len(subject_dict)>0:
                    exist_sub_classes=torch.stack(list(subject_dict.keys()))
                not_exist_sub_classes = all_classes[torch.where(torch.sum(all_classes.unsqueeze(1).repeat(1, len(exist_sub_classes)) == exist_sub_classes.unsqueeze(0).repeat(len(all_classes), 1), dim=1) == 0)]
                for sub_cls in not_exist_sub_classes:
                    not_exist_classes = all_classes
                    for obj_cls in not_exist_classes:
                        negative_sub_classes.append(sub_cls)
                        negative_obj_classes.append(obj_cls)
                negative_sub_classes=torch.stack(negative_sub_classes)
                negative_obj_classes = torch.stack(negative_obj_classes)
                randindx = torch.LongTensor(np.random.randint(0, len(negative_sub_classes),
                                                              size=min(max(5, len(to_pred_phrase_pred_subject_classes)),
                                                                       len(negative_sub_classes))))
                # print(len(to_pred_phrase_pred_subject_classes),len(randindx))
                for sub_cls,obj_cls in zip(negative_sub_classes[randindx],negative_obj_classes[randindx]):
                    to_pred_phrase_pred_subject_ids=torch.where(instance_detection.pred_classes == sub_cls)[0]
                    to_pred_phrase_pred_subject_len = torch.sum(to_pred_phrase_pred_subject_ids>=0)
                    to_pred_phrase_pred_subject_scores=torch.zeros_like(to_pred_phrase_pred_subject_ids)
                    to_pred_phrase_pred_object_ids=torch.where(instance_detection.pred_classes==obj_cls)[0]
                    to_pred_phrase_pred_object_len = torch.sum(to_pred_phrase_pred_object_ids>=0)
                    to_pred_phrase_pred_object_scores = torch.zeros_like(to_pred_phrase_pred_object_ids)
                    to_pred_phrase_pred_subject_crowd_box = union_box_matrix(
                        pred_instance_boxes_pad0[:, to_pred_phrase_pred_subject_ids])
                    to_pred_phrase_pred_object_crowd_box = union_box_matrix(
                        pred_instance_boxes_pad0[:, to_pred_phrase_pred_object_ids])
                    to_pred_phrase_pred_box = union_boxes(to_pred_phrase_pred_subject_crowd_box,
                                                          to_pred_phrase_pred_object_crowd_box)[0]
                    to_pred_phrase_pred_subject_classes.append(sub_cls)
                    to_pred_phrase_pred_object_classes.append(obj_cls)
                    to_pred_phrase_pred_subject_ids_list.append(torch.cat([to_pred_phrase_pred_subject_ids, -torch.ones(max_len - len(to_pred_phrase_pred_subject_ids)).to(self.device)]))
                    to_pred_phrase_pred_object_ids_list.append(torch.cat([to_pred_phrase_pred_object_ids, -torch.ones(max_len - len(to_pred_phrase_pred_object_ids)).to(self.device)]))
                    to_pred_phrase_pred_subject_in_clust.append(to_pred_phrase_pred_subject_ids_list[-1] >= 0)
                    to_pred_phrase_pred_object_in_clust.append(to_pred_phrase_pred_object_ids_list[-1] >= 0)
                    to_pred_phrase_pred_subject_lens.append(to_pred_phrase_pred_subject_len)
                    to_pred_phrase_pred_object_lens.append(to_pred_phrase_pred_object_len)
                    to_pred_phrase_pred_subject_scores_list.append(torch.cat([to_pred_phrase_pred_subject_scores, torch.zeros(max_len - len(to_pred_phrase_pred_subject_ids)).to(self.device)]))
                    to_pred_phrase_pred_object_scores_list.append(torch.cat([to_pred_phrase_pred_object_scores, torch.zeros(max_len - len(to_pred_phrase_pred_object_ids)).to(self.device)]))
                    to_pred_phrase_pred_boxes.append(to_pred_phrase_pred_box)
                    to_pred_phrase_pred_relation_onehots_list.append(torch.cat([torch.FloatTensor([1]).to(self.device),
                                                                     torch.zeros(self.relation_classes).to(self.device)]))
                    to_pred_phrase_confidence.append(torch.Tensor([0])[0].to(self.device))

                to_pred_phrase.pred_subject_classes = torch.stack(to_pred_phrase_pred_subject_classes)
                to_pred_phrase.pred_object_classes = torch.stack(to_pred_phrase_pred_object_classes)
                to_pred_phrase.pred_subject_ids_list = torch.stack(to_pred_phrase_pred_subject_ids_list).long()
                to_pred_phrase.pred_object_ids_list = torch.stack(to_pred_phrase_pred_object_ids_list).long()
                to_pred_phrase.pred_subject_in_clust=torch.stack(to_pred_phrase_pred_subject_in_clust)
                to_pred_phrase.pred_object_in_clust = torch.stack(to_pred_phrase_pred_object_in_clust)
                to_pred_phrase.pred_subject_lens = torch.stack(to_pred_phrase_pred_subject_lens)
                to_pred_phrase.pred_object_lens = torch.stack(to_pred_phrase_pred_object_lens)
                to_pred_phrase.pred_subject_gt_in_crowds_list = torch.stack(to_pred_phrase_pred_subject_scores_list)
                to_pred_phrase.pred_object_gt_in_crowds_list = torch.stack(to_pred_phrase_pred_object_scores_list)
                to_pred_phrase.gt_relation_onehots = torch.stack(to_pred_phrase_pred_relation_onehots_list)
                to_pred_phrase.pred_boxes = Boxes(torch.stack(to_pred_phrase_pred_boxes))
                to_pred_phrase.confidence = torch.stack(to_pred_phrase_confidence)
            to_pred_phrases.append(to_pred_phrase)

        return to_pred_phrases

    def get_to_pred_crowds_from_instance_pred_with_similarity(self, mask_preds, keeps, instance_detections, instance_similarities):
        to_pred_crowds = []
        for mask_pred, keep, instance_detection,instance_similarity in zip(mask_preds, keeps, instance_detections,instance_similarities):
            pred_instance_boxes_pad0 = torch.cat([instance_detection.pred_boxes.tensor, torch.Tensor(
                [[instance_detection.image_size[1], instance_detection.image_size[0], 0, 0]]).to(self.device)])
            instance_similarity = instance_similarity - torch.eye(len(instance_detection)).to(self.device)
            class_set = torch.unique(instance_detection.pred_classes)
            left_instance_classes = instance_detection.pred_classes.unsqueeze(1).repeat(1, len(instance_detection))
            right_instance_classes = instance_detection.pred_classes.unsqueeze(0).repeat(len(instance_detection), 1)
            sameclass = (left_instance_classes == right_instance_classes).int()
            sameclass[sameclass == 0] = -1
            class_matrix = (left_instance_classes + 1) * (sameclass)
            class_matrix[class_matrix < 0] = -1
            class_matrix = class_matrix - 1

            to_pred_crowd_instance_ids_list = []
            max_len = 1
            for thr in self.similarity:  # [0.8,0.5,0.25,0.05,0]:
                # print((instance_similarity >= thr).int())
                for ins_cls in class_set:
                    cls_instances = (class_matrix == torch.ones(len(instance_detection), len(instance_detection)).to(
                        self.device) * ins_cls).int()
                    # if thr==0.5:
                    #     print(torch.where(cls_instances==1))
                    crowd_matrix = (instance_similarity >= thr) * cls_instances
                    crowd = self.components(crowd_matrix)
                    # crowd=torch.unique(torch.where(crowd_matrix==1)[0])
                    if len(crowd) > 0:
                        # print(crowd)
                        to_pred_crowd_instance_ids_list.extend(crowd)
                        for c in crowd:
                            max_len = max(max_len, len(c))

            exist_instance_ids = []
            for i in range(len(to_pred_crowd_instance_ids_list)):
                to_pred_crowd_instance_ids_list[i] = torch.cat([to_pred_crowd_instance_ids_list[i], torch.ones(
                    max_len - len(to_pred_crowd_instance_ids_list[i])).to(self.device) * (-1)])
                exist_instance_ids.append(to_pred_crowd_instance_ids_list[i])
            if len(exist_instance_ids) > 0:
                exist_instance_ids = torch.unique(torch.cat(exist_instance_ids))
                noexist = set(range(0, len(instance_detection))).difference(set(exist_instance_ids.data.cpu().numpy().tolist()))
            else:
                noexist = set(range(0, len(instance_detection)))
            for i in noexist:
                instance_id_tensor = torch.Tensor([i]).to(self.device)
                none_instance_ids_tensor = torch.ones(max_len - 1).to(self.device) * (-1)
                to_pred_crowd_instance_ids_list.append(torch.cat([instance_id_tensor, none_instance_ids_tensor]))
            # print(to_pred_crowd_instance_ids_list)
            to_pred_crowd_instance_ids_list = torch.stack(to_pred_crowd_instance_ids_list).long()
            to_pred_crowd_instance_ids_list = torch.unique(to_pred_crowd_instance_ids_list, dim=0)
            # print(to_pred_crowd_instance_ids_list)
            # exit()
            to_pred_crowd = Instances(instance_detection.image_size)
            to_pred_crowd.pred_instance_ids_list = torch.sort(to_pred_crowd_instance_ids_list, dim=1, descending=True)[0]
            to_pred_crowd.pred_instance_lens = torch.sum(to_pred_crowd.pred_instance_ids_list >= 0, dim=1)
            to_pred_crowd.pred_boxes = Boxes(union_box_matrix(pred_instance_boxes_pad0[to_pred_crowd.pred_instance_ids_list]))
            to_pred_crowd.pred_classes = instance_detection.pred_classes[to_pred_crowd.pred_instance_ids_list[:, 0]].long()
            # to_pred_crowd.pred_in_clust = torch.ones_like(to_pred_crowd.pred_instance_ids_list)
            to_pred_crowds.append(to_pred_crowd)
        return to_pred_crowds

    def generate_to_pred_phrases_from_to_pred_crowds(self,pred_instances,to_pred_crowds):
        to_pred_phrases=[]
        af_to_pred_crowds=[]
        for image_i in range(len(pred_instances)):
            pred_instance = pred_instances[image_i]
            to_pred_crowd = to_pred_crowds[image_i]

            to_pred_crowd, to_pred_phrase = self.single_area_generate_to_pred_phrases_from_to_pred_crowds(pred_instance,to_pred_crowd)
            to_pred_phrases.append(to_pred_phrase)
            af_to_pred_crowds.append(to_pred_crowd)
        return af_to_pred_crowds, to_pred_phrases

    def single_area_generate_to_pred_phrases_from_to_pred_crowds(self,pred_instance,to_pred_crowd):
        to_pred_crowd_num = len(to_pred_crowd)
        to_pred_phrase=Instances(pred_instance.image_size)

        if to_pred_crowd_num > 0:
            to_pred_crowd_ids = torch.Tensor(range(to_pred_crowd_num)).to(self.device).long()
            to_pred_phrase_pred_subject_crowd_ids = to_pred_crowd_ids.unsqueeze(1).repeat(1, to_pred_crowd_num).flatten()
            to_pred_phrase_pred_object_crowd_ids = to_pred_crowd_ids.unsqueeze(0).repeat(to_pred_crowd_num, 1).flatten()

            to_pred_phrase.pred_subject_ids_list = to_pred_crowd.pred_instance_ids_list[to_pred_phrase_pred_subject_crowd_ids].long()
            to_pred_phrase.pred_object_ids_list = to_pred_crowd.pred_instance_ids_list[to_pred_phrase_pred_object_crowd_ids].long()
            to_pred_phrase.pred_subject_classes = to_pred_crowd.pred_classes[to_pred_phrase_pred_subject_crowd_ids].long()
            to_pred_phrase.pred_object_classes = to_pred_crowd.pred_classes[to_pred_phrase_pred_object_crowd_ids].long()
            # to_pred_phrase.pred_subject_crowd_ids = to_pred_phrase_pred_subject_crowd_ids.long()
            # to_pred_phrase.pred_object_crowd_ids = to_pred_phrase_pred_object_crowd_ids.long()
            to_pred_phrase.pred_subject_ids_list = to_pred_crowd.pred_instance_ids_list[to_pred_phrase_pred_subject_crowd_ids].long()
            to_pred_phrase.pred_object_ids_list = to_pred_crowd.pred_instance_ids_list[to_pred_phrase_pred_object_crowd_ids].long()
            to_pred_phrase.pred_subject_lens = to_pred_crowd.pred_instance_lens[to_pred_phrase_pred_subject_crowd_ids].long()
            to_pred_phrase.pred_object_lens = to_pred_crowd.pred_instance_lens[to_pred_phrase_pred_object_crowd_ids].long()
            to_pred_phrase.pred_subject_boxes = Boxes(to_pred_crowd.pred_boxes.tensor[to_pred_phrase_pred_subject_crowd_ids])
            to_pred_phrase.pred_object_boxes = Boxes(to_pred_crowd.pred_boxes.tensor[to_pred_phrase_pred_object_crowd_ids])
            # to_pred_phrase.pred_subject_in_clust  = to_pred_crowd.pred_in_clust[to_pred_phrase_pred_subject_crowd_ids]
            # to_pred_phrase.pred_object_in_clust = to_pred_crowd.pred_in_clust[to_pred_phrase_pred_object_crowd_ids]
            to_pred_phrase.pred_boxes = Boxes(union_boxes(to_pred_phrase.pred_subject_boxes.tensor, to_pred_phrase.pred_object_boxes.tensor))

        else:
            # to_pred_phrase.pred_subject_crowd_ids=torch.Tensor().to(self.device)
            # to_pred_phrase.pred_object_crowd_ids=torch.Tensor().to(self.device)
            to_pred_phrase.pred_subject_classes=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_object_classes=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_subject_ids_list=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_object_ids_list=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_subject_lens=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_object_lens=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_subject_boxes=torch.Tensor().to(self.device)
            to_pred_phrase.pred_object_boxes=torch.Tensor().to(self.device)
            to_pred_phrase.pred_boxes=Boxes(torch.Tensor().to(self.device))
            # to_pred_phrase.pred_subject_in_clust = torch.Tensor().to(self.device)
            # to_pred_phrase.pred_object_in_clust = torch.Tensor().to(self.device)

        return to_pred_crowd, to_pred_phrase

    def components(self,matrix):
        matrix=matrix.data.cpu().numpy()
        graph={}
        for i in range(len(matrix)):
            graph[i]=set(np.where(matrix[i]>0)[0].tolist())
        component = []
        if "connect" in self.graph_search_mode:
            connect_seen = set()
            for u in graph:
                if u in connect_seen:
                    continue
                current = self.walk(graph, u,mode="connect")
                connect_seen.update(current)
                if len(current)>1:
                    component.append(torch.Tensor(list(current)).to(self.device))
        if "complex" in self.graph_search_mode:
            complex_seen = set()
            for u in graph:
                if u in complex_seen:
                    continue
                current = self.walk(graph, u,mode="complex")
                complex_seen.update(current)
                if len(current)>1:
                    component.append(torch.Tensor(list(current)).to(self.device))
        return component

    def walk(self, graph, start,mode="connect"):
        nodes = set()
        # current = dict()
        seen=set([start])
        nodes.add(start)
        while nodes:
            u = nodes.pop()
            for v in graph[u].difference(seen):
                if mode == "complex":
                    if len(seen.difference(graph[v])) == 0:
                        nodes.add(v)
                        seen.add(v)
                elif mode=="connect":
                    nodes.add(v)
                    seen.add(v)
        return seen

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes, mode="instance"):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({mode + "s": r})
        return processed_results

    def inference(self, box_cls, box_pred, mask_pred, box_feature, image_sizes, instance_threshold=0.5):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        keeps = []
        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, box_feature_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, box_feature, image_sizes
        )):
            keep=scores_per_image>instance_threshold
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            # result.pred_features = box_feature_per_image
            result=result[keep]
            keeps.append(keep)
            results.append(result)
        return results, keeps

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def binary_focal_loss(self, pred, gt, pos_gamma=1.0, neg_gamma=2.0):
        # print("======================================")
        num_1 = torch.sum(gt).item() * 1.0
        num_0 = gt.shape[0] - num_1
        alpha = 0.5  # 1.0-num_1/gt.shape[0]
        # print(alpha)
        # print(pred)
        # print(gt)
        epsilon = 1.e-5
        pred = pred.clamp(epsilon, 1 - epsilon)
        ce_1 = gt * (-torch.log(pred))  # gt=1
        ce_0 = (1 - gt) * (-torch.log(1 - pred))  # gt=0

        # ce=ce_1+ce_0
        # ce_avg=torch.mean(ce)
        # print(ce_0.shape)
        # print("ce_1")
        # print(ce_1)
        # print("ce_0")
        # print(ce_0)

        fl_1 = torch.pow(1 - pred, pos_gamma) * ce_1
        # print("fl_1")
        # print(fl_1)
        # fl_1 = alpha*fl_1
        # print(fl_1)

        fl_0 = torch.pow(pred, neg_gamma) * ce_0
        # print("fl_0")
        # print(fl_0)
        # fl_0 = (1-alpha)*fl_0
        # print(fl_0)

        if num_1 == 0:
            fl_1_avg = torch.sum(fl_1)
        else:
            fl_1_avg = torch.sum(fl_1) / num_1
        if num_0 == 0:
            fl_0_avg = torch.sum(fl_0)
        else:
            fl_0_avg = torch.sum(fl_0) / num_0
        # fl=fl_0+fl_1
        # fl_avg=torch.mean(fl)
        # print(fl_avg)
        # print(fl_1_avg)
        # print(fl_0_avg)
        # print("======================================")
        return fl_1_avg, fl_0_avg

@META_ARCH_REGISTRY.register()
class DetrRelationByTransInstanceCOSSimilarity(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.relation_classes = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        self.similarity=cfg.SIMILARITY
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        self.num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        self.detr = DETR(
            backbone, transformer, num_classes=self.num_classes, num_queries=self.num_queries, aux_loss=deep_supervision
        )
        if self.mask_on:
            frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
            if frozen_weights != '':
                print("LOAD pre-trained weights")
                weight = torch.load(frozen_weights, map_location=lambda storage, loc: storage)['model']
                new_weight = {}
                for k, v in weight.items():
                    if 'detr.' in k:
                        new_weight[k.replace('detr.', '')] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.detr.load_state_dict(new_weight)
                del new_weight
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ''))
            self.seg_postprocess = PostProcessSegm

        # building criterion
        matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        if self.mask_on:
            losses += ["masks"]
        self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.instance_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

        self.instance_query_embed = nn.Embedding(1, 128)
        self.pos_encoder = nn.Linear(hidden_dim, 128)
        decoder_layer = TransformerDecoderLayer(
            d_model=128,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            normalize_before=pre_norm)
        decoder_norm = nn.LayerNorm(128)
        self.instance_mask_decoder = TransformerDecoder(decoder_layer, dec_layers, decoder_norm,return_intermediate=deep_supervision)
        self.instance_encoder = nn.Linear(hidden_dim + 128, 512)
        self.location_dim=2 #cfg.MODEL.ROI_HEADS.LOCATION_DIM
        self.location_encode = nn.Sequential(nn.Linear(self.location_dim, 256),nn.Linear(256, 512))

        self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST
        self.relation_heads = build_relation_heads(cfg)
        # self.positive_crowd_threshold = cfg.MODEL.ROI_HEADS.POSITIVE_CROWD_THRESHOLD
        # self.positive_box_threshold = cfg.MODEL.ROI_HEADS.POSITIVE_BOX_THRESHOLD

        # self.gt_similarity_mode = cfg.MODEL.ROI_HEADS.GT_SIMILARITY_MODE
        self.similarity_mode = cfg.MODEL.ROI_HEADS.SIMILARITY_MODE
        self.graph_search_mode = cfg.MODEL.ROI_HEADS.GRAPH_SEARCH_MODE
        #
        # self.similarity_loss = cfg.MODEL.ROI_HEADS.SIMILARITY_LOSS
        # self.train_only_same_class = cfg.MODEL.ROI_HEADS.SIMILARITY_TRAIN_ONLY_SAME_CLASS

        self.to(self.device)

    def forward(self, batched_inputs,iteration, mode, training, print_size=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        self.print_size=print_size
        losses = {}
        results=[]
        metrics={}
        relation_results=None

        images = self.preprocess_image(batched_inputs)

        output, features, pos = self.detr(images)
        src, mask = features[-1].decompose()
        bs, c, h, w = src.shape

        if mode=="instance":
            if not training:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else None
                box_feature = output["pred_features"]
                instance_detections,keeps = self.inference(box_cls, box_pred, mask_pred, box_feature, images.image_sizes)
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes,
                                                     "instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(instance_results[i])
            else:
                print("not support instance training")
                exit()
        elif mode=="relation":
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            box_feature = output["pred_features"][-1]
            mask_feature = output["mask_features"]
            instance_detections, keeps = self.inference(box_cls, box_pred, mask_pred, box_feature, images.image_sizes,
                                                        self.instance_threshold)
            # self.fullize_mask(instance_detections)
            mask_pred = F.sigmoid(mask_pred)

            pos_embed = pos[-1].unsqueeze(1).repeat(1, self.num_queries, 1, 1, 1)
            query_embed = self.instance_query_embed.weight.unsqueeze(1).repeat(1, bs * self.num_queries, 1)
            tgt = torch.zeros_like(query_embed)
            # print("tgt",tgt.shape)
            # print("memory",g_mask_feature.permute(3,4,0,1,2).flatten(0,1).flatten(1,2).shape)
            # print("mask",mask.unsqueeze(1).repeat(1,int(self.num_queries/5),1,1).flatten(0,1).flatten(1,2).shape)
            pos_embed = pos_embed.flatten(0, 1).flatten(2).permute(2, 0, 1)
            pos_embed_encoded = self.pos_encoder(pos_embed)
            # print("pos_embed", pos_embed_encoded.shape)
            # print("query_embed", query_embed.shape)
            hs = self.instance_mask_decoder(tgt,
                                            mask_feature.permute(3, 4, 0, 1, 2).flatten(0, 1).flatten(1, 2),
                                            # 1,20,8,10,36 --> 10,36,1,20,8 --> 360,20,8
                                            memory_key_padding_mask=mask.unsqueeze(1).repeat(1, self.num_queries, 1,1).flatten(0, 1).flatten(1,2),
                                            # 1,10,36 --> 1,20,10,36 -> 20,360
                                            pos=pos_embed_encoded,
                                            query_pos=query_embed)
            # print("hs",hs.shape)
            instance_mask_features = hs[-1].squeeze(1).view(bs, self.num_queries, hs.shape[-1])

            instance_location_features = []
            for pred_instance in instance_detections:
                if self.location_dim == 4:
                    global_location = torch.Tensor([0, 0, pred_instance.image_size[1], pred_instance.image_size[0]]).to(self.device)
                    global_location = global_location.unsqueeze(0).repeat(len(pred_instance), 1)
                    instance_location_feature = pred_instance.pred_boxes.tensor - global_location  # x1,y1,x2,y2
                    global_location = torch.Tensor([pred_instance.image_size[1], pred_instance.image_size[0], pred_instance.image_size[1],pred_instance.image_size[0]]).to(self.device)
                    global_location = global_location.unsqueeze(0).repeat(len(pred_instance), 1)
                    instance_location_feature = instance_location_feature / global_location
                elif self.location_dim == 2:
                    instance_location_feature = pred_instance.pred_boxes.tensor[:,2:] - pred_instance.pred_boxes.tensor[:, :2]  # x1,y1,x2,y2
                    global_location = torch.Tensor([pred_instance.image_size[1], pred_instance.image_size[0]]).to(self.device)
                    global_location = global_location.unsqueeze(0).repeat(len(pred_instance), 1)
                    instance_location_feature = instance_location_feature / global_location
                instance_location_features.append(self.location_encode(instance_location_feature))

            instance_local_features = [self.instance_encoder(torch.cat([b_feature[keep], m_feature[keep]], dim=1)) for
                                       b_feature, m_feature, keep in zip(box_feature, instance_mask_features, keeps)]

            if training:
                gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
                gt_phrases = [x["phrases"].to(self.device) for x in batched_inputs]

                g_mask_gts = []
                for mask_pred_single, gt_crowd in zip(mask_pred, gt_crowds):
                    pred_mask = mask_pred_single
                    gt_mask = gt_crowd.gt_masks.tensor
                    gt_mask_reshape = F.interpolate(gt_mask.unsqueeze(0).float(), size=pred_mask.shape[1:],
                                                    mode='bilinear',
                                                    align_corners=False)
                    g_mask_gts.append(gt_mask_reshape[0])

                    # if self.similarity_mode=="att":
                    #     instance_similarities.append(attention)
                    # elif self.similarity_mode=="distance":
                    #     euclidean_distance = self.euclidean_dist(prime, prime)
                    #     instance_similarities.append(torch.exp(-euclidean_distance))
                    # elif self.similarity_mode=="cosine":
                    #     euclidean_distance = F.cosine_similarity(prime.unsqueeze(1).repeat(1,prime.shape[0],1).view(prime.shape[0]*prime.shape[0],prime.shape[1]),
                    #                                              prime.unsqueeze(0).repeat(prime.shape[0],1,1).view(prime.shape[0]*prime.shape[0],prime.shape[1]))
                    #     euclidean_distance=euclidean_distance.view(prime.shape[0],prime.shape[0])
                    #     instance_similarities.append((euclidean_distance+1)/2)
                    # else:
                    #     print("no such similarity function: " + self.similarity_mode)
                    #     exit()

                # if training:
                #     gt_instance_similarities = self.gt_similarity(instance_detections, gt_instances, gt_crowds)
                #     if self.train_only_same_class:
                #         gts, preds = [], []
                #         for pred_instance, gt_instance_similarity, instance_similarity in zip(instance_detections,
                #                                                                               gt_instance_similarities,
                #                                                                               instance_similarities):
                #             label_marix = pred_instance.pred_classes.unsqueeze(1).repeat(1, len(pred_instance))
                #             label_marix_1 = pred_instance.pred_classes.unsqueeze(0).repeat(len(pred_instance), 1)
                #             choose_index = (label_marix == label_marix_1)
                #             gts.append(gt_instance_similarity[choose_index])
                #             preds.append(instance_similarity[choose_index])
                #
                #         gts = torch.cat(gts)
                #         preds = torch.cat(preds)
                #     else:
                #         gts = torch.cat(gt_instance_similarities)
                #         preds = torch.cat(instance_similarities)
                #     print(class_similarity_score)
                #     print(gts)
                #     print(preds)
                #     print('------------------------------------------------------')
                #
                #     if self.similarity_loss == "f1":
                #         losses['loss_similarity'] = F.l1_loss(preds, gts)
                #     elif self.similarity_loss == "smooth_f1":
                #         losses['loss_similarity'] = F.smooth_l1_loss(preds, gts)
                #     elif self.similarity_loss == "bi_focal":
                #         losses['loss_pos_similarity'], losses['loss_neg_similarity'] = self.binary_focal_loss(preds,gts)

                if len(instance_detections[0]) > 0:

                    to_pred_phrases = self.get_target_from_instance_pred(mask_pred, keeps, instance_detections,g_mask_gts, gt_phrases)

                    if len(to_pred_phrases[0]) > 0:
                        relation_results, rela_losses, rela_metrics = self.relation_heads(instance_detections,
                                                                                          instance_local_features,
                                                                                          to_pred_phrases,
                                                                                          training=training,
                                                                                          iteration=iteration)
                        losses.update(rela_losses)
                    else:
                        losses = {}
                else:
                    losses = {}
            else:
                # instance_local_features_pad0=[]
                instance_similarities = []
                for pred_instance, instance_local_feature, instance_location_feature in zip(instance_detections,
                                                                                            instance_local_features,
                                                                                            instance_location_features):
                    instance_feature = torch.cat([instance_local_feature, instance_location_feature], dim=1)
                    class_similarity_score = (pred_instance.pred_classes.unsqueeze(1).repeat(1, len(pred_instance)) == pred_instance.pred_classes.unsqueeze(0).repeat(len(pred_instance),1)).float()
                    # prime,attention=self.gat(instance_feature,class_similarity_score)
                    simi_flatten = F.cosine_similarity(
                        instance_feature.unsqueeze(1).repeat(1, instance_feature.shape[0], 1).view(instance_feature.shape[0] * instance_feature.shape[0], instance_feature.shape[1]),
                        instance_feature.unsqueeze(0).repeat(instance_feature.shape[0], 1, 1).view(instance_feature.shape[0] * instance_feature.shape[0], instance_feature.shape[1]))
                    instance_similarities.append((simi_flatten.view(instance_feature.shape[0], instance_feature.shape[0]) + 1) / 2)
                    # print(class_similarity_score)
                    # print(instance_similarities)
                    # exit()
                if len(instance_detections[0]) > 0:
                    to_pred_crowds = self.get_to_pred_crowds_from_instance_pred_with_similarity(mask_pred, keeps, instance_detections, instance_similarities)
                    to_pred_crowds, to_pred_phrases = self.generate_to_pred_phrases_from_to_pred_crowds(instance_detections, to_pred_crowds)
                    if len(to_pred_phrases[0]) > 0:
                        relation_results, rela_losses, rela_metrics = self.relation_heads(instance_detections,
                                                                                          instance_local_features,
                                                                                          to_pred_phrases,
                                                                                          training=training,
                                                                                          iteration=iteration)
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes,"instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(instance_results[i])
                    if relation_results:
                        results[i]['phrases']=relation_results[i]

        return results, losses, metrics

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks.tensor
                # gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({'masks': gt_masks})
        return new_targets

    def fullize_mask(self, instances):
        for instance_per_image in instances:
            height, width = instance_per_image.image_size
            fullize_mask(instance_per_image, height, width)

    def get_target_from_instance_pred(self,mask_preds, keeps, instance_detections, g_mask_gts, gt_phrases):
        to_pred_phrases=[]
        for mask_pred, keep, instance_detection, g_mask_gt, gt_phrase in zip(mask_preds, keeps, instance_detections, g_mask_gts, gt_phrases):
            pred_instance_boxes_pad0 = torch.cat([instance_detection.pred_boxes.tensor, torch.Tensor([[instance_detection.image_size[1], instance_detection.image_size[0], 0, 0]]).to(self.device)]).unsqueeze(0)
            to_pred_phrase=Instances(image_size=instance_detection.image_size)
            mask_pred=mask_pred[keep]>0.5
            g_mask_gt=g_mask_gt>0.5
            # phrase, instance
            instance_area = torch.sum(mask_pred.unsqueeze(0).repeat(len(gt_phrase), 1, 1, 1), dim=[2, 3])
            subject_interaction_area=torch.sum(mask_pred.unsqueeze(0).repeat(len(gt_phrase), 1, 1, 1)
                                               * g_mask_gt[gt_phrase.gt_subject_crowd_ids].unsqueeze(1).repeat(1, len(instance_detection), 1, 1), dim=[2, 3])
            gt_phrase_subject_instance_match_score = subject_interaction_area/instance_area \
                                                     * (instance_detection.pred_classes.unsqueeze(0).repeat(len(gt_phrase), 1)
                                                        ==gt_phrase.gt_subject_classes.unsqueeze(1).repeat(1, len(instance_detection)))
            object_interaction_area = torch.sum(mask_pred.unsqueeze(0).repeat(len(gt_phrase), 1, 1, 1) \
                                                * g_mask_gt[gt_phrase.gt_object_crowd_ids].unsqueeze(1).repeat(1, len(instance_detection), 1, 1), dim=[2, 3])
            gt_phrase_object_instance_match_score = object_interaction_area / instance_area \
                                             * (instance_detection.pred_classes.unsqueeze(0).repeat(len(gt_phrase), 1)
                                                == gt_phrase.gt_object_classes.unsqueeze(1).repeat(1, len(instance_detection)))
            subject_dict=defaultdict(list)
            object_dict = defaultdict(list)
            all_classes=torch.unique(instance_detection.pred_classes)
            max_len=0
            for cls in all_classes:
                cls_length=torch.sum(instance_detection.pred_classes==cls).item()
                max_len=max(cls_length,max_len)
            subject_classes = gt_phrase.gt_subject_classes
            object_classes = gt_phrase.gt_object_classes
            to_pred_phrase_pred_subject_lens = []
            to_pred_phrase_pred_object_lens = []
            to_pred_phrase_pred_boxes = []
            to_pred_phrase_pred_subject_classes = []
            to_pred_phrase_pred_object_classes = []
            to_pred_phrase_pred_subject_ids_list=[]
            to_pred_phrase_pred_object_ids_list = []
            to_pred_phrase_pred_subject_in_clust=[]
            to_pred_phrase_pred_object_in_clust = []
            to_pred_phrase_pred_subject_scores_list=[]
            to_pred_phrase_pred_object_scores_list = []
            to_pred_phrase_pred_relation_onehots_list=[]
            to_pred_phrase_confidence=[]
            for i,(single_gt_phrase_subject_instance_match_score,single_gt_phrase_object_instance_match_score) in enumerate(zip(gt_phrase_subject_instance_match_score,gt_phrase_object_instance_match_score)):
                to_pred_phrase_pred_subject_ids = torch.where(single_gt_phrase_subject_instance_match_score>0)[0]
                to_pred_phrase_pred_object_ids = torch.where(single_gt_phrase_object_instance_match_score > 0)[0]
                to_pred_phrase_pred_subject_len = torch.sum(to_pred_phrase_pred_subject_ids>=0)
                to_pred_phrase_pred_object_len = torch.sum(to_pred_phrase_pred_object_ids>=0)
                to_pred_phrase_pred_subject_scores = single_gt_phrase_subject_instance_match_score[to_pred_phrase_pred_subject_ids]
                to_pred_phrase_pred_object_scores = single_gt_phrase_object_instance_match_score[to_pred_phrase_pred_object_ids]
                if to_pred_phrase_pred_subject_len.item()<=0 or to_pred_phrase_pred_object_len.item()<=0:
                    continue
                to_pred_phrase_pred_subject_crowd_box=union_box_matrix(pred_instance_boxes_pad0[:,to_pred_phrase_pred_subject_ids])
                to_pred_phrase_pred_object_crowd_box = union_box_matrix(pred_instance_boxes_pad0[:,to_pred_phrase_pred_object_ids])
                to_pred_phrase_pred_box=union_boxes(to_pred_phrase_pred_subject_crowd_box,
                                                        to_pred_phrase_pred_object_crowd_box)[0]
                to_pred_phrase_pred_subject_classes.append(gt_phrase.gt_subject_classes[i])
                to_pred_phrase_pred_object_classes.append(gt_phrase.gt_object_classes[i])
                to_pred_phrase_pred_subject_ids_list.append(torch.cat([to_pred_phrase_pred_subject_ids,-torch.ones(max_len-len(to_pred_phrase_pred_subject_ids)).to(self.device)]))
                to_pred_phrase_pred_object_ids_list.append(torch.cat([to_pred_phrase_pred_object_ids,-torch.ones(max_len-len(to_pred_phrase_pred_object_ids)).to(self.device)]))
                to_pred_phrase_pred_subject_in_clust.append(to_pred_phrase_pred_subject_ids_list[-1]>=0)
                to_pred_phrase_pred_object_in_clust.append(to_pred_phrase_pred_object_ids_list[-1] >= 0)
                to_pred_phrase_pred_subject_lens.append(to_pred_phrase_pred_subject_len)
                to_pred_phrase_pred_object_lens.append(to_pred_phrase_pred_object_len)
                to_pred_phrase_pred_subject_scores_list.append(torch.cat([to_pred_phrase_pred_subject_scores,torch.zeros(max_len-len(to_pred_phrase_pred_subject_ids)).to(self.device)]))
                to_pred_phrase_pred_object_scores_list.append(torch.cat([to_pred_phrase_pred_object_scores,torch.zeros(max_len-len(to_pred_phrase_pred_object_ids)).to(self.device)]))
                to_pred_phrase_pred_boxes.append(to_pred_phrase_pred_box)
                to_pred_phrase_pred_relation_onehots_list.append(gt_phrase.gt_relation_onehots[i])
                to_pred_phrase_confidence.append(1-gt_phrase.gt_relation_onehots[i][0])
                subject_dict[subject_classes[i]].append(object_classes[i])
                object_dict[object_classes[i]].append(subject_classes[i])

            if len(to_pred_phrase_pred_subject_classes)>0:
                negative_sub_classes=[]
                negative_obj_classes = []
                for sub_cls in subject_dict:
                    exist_classes=torch.stack(subject_dict[sub_cls])
                    not_exist_classes=all_classes[torch.where(torch.sum(all_classes.unsqueeze(1).repeat(1,len(exist_classes))==exist_classes.unsqueeze(0).repeat(len(all_classes),1),dim=1)==0)]
                    for obj_cls in not_exist_classes:
                        negative_sub_classes.append(sub_cls)
                        negative_obj_classes.append(obj_cls)

                exist_sub_classes = torch.Tensor([]).to(self.device)
                if len(subject_dict)>0:
                    exist_sub_classes=torch.stack(list(subject_dict.keys()))
                not_exist_sub_classes = all_classes[torch.where(torch.sum(all_classes.unsqueeze(1).repeat(1, len(exist_sub_classes)) == exist_sub_classes.unsqueeze(0).repeat(len(all_classes), 1), dim=1) == 0)]
                for sub_cls in not_exist_sub_classes:
                    not_exist_classes = all_classes
                    for obj_cls in not_exist_classes:
                        negative_sub_classes.append(sub_cls)
                        negative_obj_classes.append(obj_cls)
                negative_sub_classes=torch.stack(negative_sub_classes)
                negative_obj_classes = torch.stack(negative_obj_classes)
                randindx = torch.LongTensor(np.random.randint(0, len(negative_sub_classes),
                                                              size=min(max(5, len(to_pred_phrase_pred_subject_classes)),
                                                                       len(negative_sub_classes))))
                # print(len(to_pred_phrase_pred_subject_classes),len(randindx))
                for sub_cls,obj_cls in zip(negative_sub_classes[randindx],negative_obj_classes[randindx]):
                    to_pred_phrase_pred_subject_ids=torch.where(instance_detection.pred_classes == sub_cls)[0]
                    to_pred_phrase_pred_subject_len = torch.sum(to_pred_phrase_pred_subject_ids>=0)
                    to_pred_phrase_pred_subject_scores=torch.zeros_like(to_pred_phrase_pred_subject_ids)
                    to_pred_phrase_pred_object_ids=torch.where(instance_detection.pred_classes==obj_cls)[0]
                    to_pred_phrase_pred_object_len = torch.sum(to_pred_phrase_pred_object_ids>=0)
                    to_pred_phrase_pred_object_scores = torch.zeros_like(to_pred_phrase_pred_object_ids)
                    to_pred_phrase_pred_subject_crowd_box = union_box_matrix(
                        pred_instance_boxes_pad0[:, to_pred_phrase_pred_subject_ids])
                    to_pred_phrase_pred_object_crowd_box = union_box_matrix(
                        pred_instance_boxes_pad0[:, to_pred_phrase_pred_object_ids])
                    to_pred_phrase_pred_box = union_boxes(to_pred_phrase_pred_subject_crowd_box,
                                                          to_pred_phrase_pred_object_crowd_box)[0]
                    to_pred_phrase_pred_subject_classes.append(sub_cls)
                    to_pred_phrase_pred_object_classes.append(obj_cls)
                    to_pred_phrase_pred_subject_ids_list.append(torch.cat([to_pred_phrase_pred_subject_ids, -torch.ones(max_len - len(to_pred_phrase_pred_subject_ids)).to(self.device)]))
                    to_pred_phrase_pred_object_ids_list.append(torch.cat([to_pred_phrase_pred_object_ids, -torch.ones(max_len - len(to_pred_phrase_pred_object_ids)).to(self.device)]))
                    to_pred_phrase_pred_subject_in_clust.append(to_pred_phrase_pred_subject_ids_list[-1] >= 0)
                    to_pred_phrase_pred_object_in_clust.append(to_pred_phrase_pred_object_ids_list[-1] >= 0)
                    to_pred_phrase_pred_subject_lens.append(to_pred_phrase_pred_subject_len)
                    to_pred_phrase_pred_object_lens.append(to_pred_phrase_pred_object_len)
                    to_pred_phrase_pred_subject_scores_list.append(torch.cat([to_pred_phrase_pred_subject_scores, torch.zeros(max_len - len(to_pred_phrase_pred_subject_ids)).to(self.device)]))
                    to_pred_phrase_pred_object_scores_list.append(torch.cat([to_pred_phrase_pred_object_scores, torch.zeros(max_len - len(to_pred_phrase_pred_object_ids)).to(self.device)]))
                    to_pred_phrase_pred_boxes.append(to_pred_phrase_pred_box)
                    to_pred_phrase_pred_relation_onehots_list.append(torch.cat([torch.FloatTensor([1]).to(self.device),
                                                                     torch.zeros(self.relation_classes).to(self.device)]))
                    to_pred_phrase_confidence.append(1-gt_phrase.gt_relation_onehots[i][0])

                to_pred_phrase.pred_subject_classes = torch.stack(to_pred_phrase_pred_subject_classes)
                to_pred_phrase.pred_object_classes = torch.stack(to_pred_phrase_pred_object_classes)
                to_pred_phrase.pred_subject_ids_list = torch.stack(to_pred_phrase_pred_subject_ids_list).long()
                to_pred_phrase.pred_object_ids_list = torch.stack(to_pred_phrase_pred_object_ids_list).long()
                to_pred_phrase.pred_subject_in_clust=torch.stack(to_pred_phrase_pred_subject_in_clust)
                to_pred_phrase.pred_object_in_clust = torch.stack(to_pred_phrase_pred_object_in_clust)
                to_pred_phrase.pred_subject_lens = torch.stack(to_pred_phrase_pred_subject_lens)
                to_pred_phrase.pred_object_lens = torch.stack(to_pred_phrase_pred_object_lens)
                to_pred_phrase.pred_subject_gt_in_crowds_list = torch.stack(to_pred_phrase_pred_subject_scores_list)
                to_pred_phrase.pred_object_gt_in_crowds_list = torch.stack(to_pred_phrase_pred_object_scores_list)
                to_pred_phrase.gt_relation_onehots = torch.stack(to_pred_phrase_pred_relation_onehots_list)
                to_pred_phrase.pred_boxes = Boxes(torch.stack(to_pred_phrase_pred_boxes))
                to_pred_phrase.confidence = torch.stack(to_pred_phrase_confidence)
            to_pred_phrases.append(to_pred_phrase)

        return to_pred_phrases

    def get_to_pred_crowds_from_instance_pred_with_similarity(self, mask_preds, keeps, instance_detections, instance_similarities):
        to_pred_crowds = []
        for mask_pred, keep, instance_detection,instance_similarity in zip(mask_preds, keeps, instance_detections,instance_similarities):
            pred_instance_boxes_pad0 = torch.cat([instance_detection.pred_boxes.tensor, torch.Tensor(
                [[instance_detection.image_size[1], instance_detection.image_size[0], 0, 0]]).to(self.device)])
            instance_similarity = instance_similarity - torch.eye(len(instance_detection)).to(self.device)
            class_set = torch.unique(instance_detection.pred_classes)
            left_instance_classes = instance_detection.pred_classes.unsqueeze(1).repeat(1, len(instance_detection))
            right_instance_classes = instance_detection.pred_classes.unsqueeze(0).repeat(len(instance_detection), 1)
            sameclass = (left_instance_classes == right_instance_classes).int()
            sameclass[sameclass == 0] = -1
            class_matrix = (left_instance_classes + 1) * (sameclass)
            class_matrix[class_matrix < 0] = -1
            class_matrix = class_matrix - 1

            to_pred_crowd_instance_ids_list = []
            max_len = 1
            for thr in self.similarity:  # [0.8,0.5,0.25,0.05,0]:
                # print((instance_similarity >= thr).int())
                for ins_cls in class_set:
                    cls_instances = (class_matrix == torch.ones(len(instance_detection), len(instance_detection)).to(
                        self.device) * ins_cls).int()
                    # if thr==0.5:
                    #     print(torch.where(cls_instances==1))
                    crowd_matrix = (instance_similarity >= thr) * cls_instances
                    crowd = self.components(crowd_matrix)
                    # crowd=torch.unique(torch.where(crowd_matrix==1)[0])
                    if len(crowd) > 0:
                        # print(crowd)
                        to_pred_crowd_instance_ids_list.extend(crowd)
                        for c in crowd:
                            max_len = max(max_len, len(c))

            exist_instance_ids = []
            for i in range(len(to_pred_crowd_instance_ids_list)):
                to_pred_crowd_instance_ids_list[i] = torch.cat([to_pred_crowd_instance_ids_list[i], torch.ones(
                    max_len - len(to_pred_crowd_instance_ids_list[i])).to(self.device) * (-1)])
                exist_instance_ids.append(to_pred_crowd_instance_ids_list[i])
            if len(exist_instance_ids) > 0:
                exist_instance_ids = torch.unique(torch.cat(exist_instance_ids))
                noexist = set(range(0, len(instance_detection))).difference(set(exist_instance_ids.data.cpu().numpy().tolist()))
            else:
                noexist = set(range(0, len(instance_detection)))
            for i in noexist:
                instance_id_tensor = torch.Tensor([i]).to(self.device)
                none_instance_ids_tensor = torch.ones(max_len - 1).to(self.device) * (-1)
                to_pred_crowd_instance_ids_list.append(torch.cat([instance_id_tensor, none_instance_ids_tensor]))
            # print(to_pred_crowd_instance_ids_list)
            to_pred_crowd_instance_ids_list = torch.stack(to_pred_crowd_instance_ids_list).long()
            to_pred_crowd_instance_ids_list = torch.unique(to_pred_crowd_instance_ids_list, dim=0)
            # print(to_pred_crowd_instance_ids_list)
            # exit()
            to_pred_crowd = Instances(instance_detection.image_size)
            to_pred_crowd.pred_instance_ids_list = torch.sort(to_pred_crowd_instance_ids_list, dim=1, descending=True)[0]
            to_pred_crowd.pred_instance_lens = torch.sum(to_pred_crowd.pred_instance_ids_list >= 0, dim=1)
            to_pred_crowd.pred_boxes = Boxes(union_box_matrix(pred_instance_boxes_pad0[to_pred_crowd.pred_instance_ids_list]))
            to_pred_crowd.pred_classes = instance_detection.pred_classes[to_pred_crowd.pred_instance_ids_list[:, 0]].long()
            # to_pred_crowd.pred_in_clust = torch.ones_like(to_pred_crowd.pred_instance_ids_list)
            to_pred_crowds.append(to_pred_crowd)
        return to_pred_crowds

    def generate_to_pred_phrases_from_to_pred_crowds(self,pred_instances,to_pred_crowds):
        to_pred_phrases=[]
        af_to_pred_crowds=[]
        for image_i in range(len(pred_instances)):
            pred_instance = pred_instances[image_i]
            to_pred_crowd = to_pred_crowds[image_i]

            to_pred_crowd, to_pred_phrase = self.single_area_generate_to_pred_phrases_from_to_pred_crowds(pred_instance,to_pred_crowd)
            to_pred_phrases.append(to_pred_phrase)
            af_to_pred_crowds.append(to_pred_crowd)
        return af_to_pred_crowds, to_pred_phrases

    def single_area_generate_to_pred_phrases_from_to_pred_crowds(self,pred_instance,to_pred_crowd):
        to_pred_crowd_num = len(to_pred_crowd)
        to_pred_phrase=Instances(pred_instance.image_size)

        if to_pred_crowd_num > 0:
            to_pred_crowd_ids = torch.Tensor(range(to_pred_crowd_num)).to(self.device).long()
            to_pred_phrase_pred_subject_crowd_ids = to_pred_crowd_ids.unsqueeze(1).repeat(1, to_pred_crowd_num).flatten()
            to_pred_phrase_pred_object_crowd_ids = to_pred_crowd_ids.unsqueeze(0).repeat(to_pred_crowd_num, 1).flatten()

            to_pred_phrase.pred_subject_ids_list = to_pred_crowd.pred_instance_ids_list[to_pred_phrase_pred_subject_crowd_ids].long()
            to_pred_phrase.pred_object_ids_list = to_pred_crowd.pred_instance_ids_list[to_pred_phrase_pred_object_crowd_ids].long()
            to_pred_phrase.pred_subject_classes = to_pred_crowd.pred_classes[to_pred_phrase_pred_subject_crowd_ids].long()
            to_pred_phrase.pred_object_classes = to_pred_crowd.pred_classes[to_pred_phrase_pred_object_crowd_ids].long()
            # to_pred_phrase.pred_subject_crowd_ids = to_pred_phrase_pred_subject_crowd_ids.long()
            # to_pred_phrase.pred_object_crowd_ids = to_pred_phrase_pred_object_crowd_ids.long()
            to_pred_phrase.pred_subject_ids_list = to_pred_crowd.pred_instance_ids_list[to_pred_phrase_pred_subject_crowd_ids].long()
            to_pred_phrase.pred_object_ids_list = to_pred_crowd.pred_instance_ids_list[to_pred_phrase_pred_object_crowd_ids].long()
            to_pred_phrase.pred_subject_lens = to_pred_crowd.pred_instance_lens[to_pred_phrase_pred_subject_crowd_ids].long()
            to_pred_phrase.pred_object_lens = to_pred_crowd.pred_instance_lens[to_pred_phrase_pred_object_crowd_ids].long()
            to_pred_phrase.pred_subject_boxes = Boxes(to_pred_crowd.pred_boxes.tensor[to_pred_phrase_pred_subject_crowd_ids])
            to_pred_phrase.pred_object_boxes = Boxes(to_pred_crowd.pred_boxes.tensor[to_pred_phrase_pred_object_crowd_ids])
            # to_pred_phrase.pred_subject_in_clust  = to_pred_crowd.pred_in_clust[to_pred_phrase_pred_subject_crowd_ids]
            # to_pred_phrase.pred_object_in_clust = to_pred_crowd.pred_in_clust[to_pred_phrase_pred_object_crowd_ids]
            to_pred_phrase.pred_boxes = Boxes(union_boxes(to_pred_phrase.pred_subject_boxes.tensor, to_pred_phrase.pred_object_boxes.tensor))

        else:
            # to_pred_phrase.pred_subject_crowd_ids=torch.Tensor().to(self.device)
            # to_pred_phrase.pred_object_crowd_ids=torch.Tensor().to(self.device)
            to_pred_phrase.pred_subject_classes=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_object_classes=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_subject_ids_list=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_object_ids_list=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_subject_lens=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_object_lens=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_subject_boxes=torch.Tensor().to(self.device)
            to_pred_phrase.pred_object_boxes=torch.Tensor().to(self.device)
            to_pred_phrase.pred_boxes=Boxes(torch.Tensor().to(self.device))
            # to_pred_phrase.pred_subject_in_clust = torch.Tensor().to(self.device)
            # to_pred_phrase.pred_object_in_clust = torch.Tensor().to(self.device)

        return to_pred_crowd, to_pred_phrase

    def components(self,matrix):
        matrix=matrix.data.cpu().numpy()
        graph={}
        for i in range(len(matrix)):
            graph[i]=set(np.where(matrix[i]>0)[0].tolist())
        component = []
        if "connect" in self.graph_search_mode:
            connect_seen = set()
            for u in graph:
                if u in connect_seen:
                    continue
                current = self.walk(graph, u,mode="connect")
                connect_seen.update(current)
                if len(current)>1:
                    component.append(torch.Tensor(list(current)).to(self.device))
        if "complex" in self.graph_search_mode:
            complex_seen = set()
            for u in graph:
                if u in complex_seen:
                    continue
                current = self.walk(graph, u,mode="complex")
                complex_seen.update(current)
                if len(current)>1:
                    component.append(torch.Tensor(list(current)).to(self.device))
        return component

    def walk(self, graph, start,mode="connect"):
        nodes = set()
        # current = dict()
        seen=set([start])
        nodes.add(start)
        while nodes:
            u = nodes.pop()
            for v in graph[u].difference(seen):
                if mode == "complex":
                    if len(seen.difference(graph[v])) == 0:
                        nodes.add(v)
                        seen.add(v)
                elif mode=="connect":
                    nodes.add(v)
                    seen.add(v)
        return seen

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes, mode="instance"):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({mode + "s": r})
        return processed_results

    def inference(self, box_cls, box_pred, mask_pred, box_feature, image_sizes, instance_threshold=0.5):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        keeps = []
        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, box_feature_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, box_feature, image_sizes
        )):
            keep=scores_per_image>instance_threshold
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            # result.pred_features = box_feature_per_image
            result=result[keep]
            keeps.append(keep)
            results.append(result)
        return results, keeps

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def binary_focal_loss(self, pred, gt, pos_gamma=1.0, neg_gamma=2.0):
        # print("======================================")
        num_1 = torch.sum(gt).item() * 1.0
        num_0 = gt.shape[0] - num_1
        alpha = 0.5  # 1.0-num_1/gt.shape[0]
        # print(alpha)
        # print(pred)
        # print(gt)
        epsilon = 1.e-5
        pred = pred.clamp(epsilon, 1 - epsilon)
        ce_1 = gt * (-torch.log(pred))  # gt=1
        ce_0 = (1 - gt) * (-torch.log(1 - pred))  # gt=0

        # ce=ce_1+ce_0
        # ce_avg=torch.mean(ce)
        # print(ce_0.shape)
        # print("ce_1")
        # print(ce_1)
        # print("ce_0")
        # print(ce_0)

        fl_1 = torch.pow(1 - pred, pos_gamma) * ce_1
        # print("fl_1")
        # print(fl_1)
        # fl_1 = alpha*fl_1
        # print(fl_1)

        fl_0 = torch.pow(pred, neg_gamma) * ce_0
        # print("fl_0")
        # print(fl_0)
        # fl_0 = (1-alpha)*fl_0
        # print(fl_0)

        if num_1 == 0:
            fl_1_avg = torch.sum(fl_1)
        else:
            fl_1_avg = torch.sum(fl_1) / num_1
        if num_0 == 0:
            fl_0_avg = torch.sum(fl_0)
        else:
            fl_0_avg = torch.sum(fl_0) / num_0
        # fl=fl_0+fl_1
        # fl_avg=torch.mean(fl)
        # print(fl_avg)
        # print(fl_1_avg)
        # print(fl_0_avg)
        # print("======================================")
        return fl_1_avg, fl_0_avg


@META_ARCH_REGISTRY.register()
class DetrRelationByCrowd(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.relation_classes = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        self.num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        gtransformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        self.detr = DETR(
            backbone, transformer, num_classes=self.num_classes, num_queries=self.num_queries, aux_loss=deep_supervision
        )
        self.gdetr = GDETR(
            backbone.num_channels, gtransformer, num_classes=self.num_classes, num_queries=int(self.num_queries/5), aux_loss=deep_supervision
        )
        if self.mask_on:
            frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
            if frozen_weights != '':
                print("LOAD pre-trained weights")
                weight = torch.load(frozen_weights, map_location=lambda storage, loc: storage)['model']
                new_weight = {}
                for k, v in weight.items():
                    if 'detr.' in k:
                        new_weight[k.replace('detr.', '')] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.detr.load_state_dict(new_weight)
                del new_weight
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ''))
            self.seg_postprocess = PostProcessSegm
            self.gdetr = GDETRsegm(self.gdetr)

        # building criterion
        matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        if self.mask_on:
            losses += ["masks"]
        self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.instance_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

        self.crowd_query_embed = nn.Embedding(1, 128)
        self.pos_encoder = nn.Linear(hidden_dim,128)
        decoder_layer = TransformerDecoderLayer(
            d_model=128,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            normalize_before=pre_norm)
        decoder_norm = nn.LayerNorm(128)
        self.crowd_mask_decoder = TransformerDecoder(decoder_layer, dec_layers, decoder_norm, return_intermediate=deep_supervision)

        self.crowd_encoder=nn.Linear(hidden_dim+128,128)
        self.crowd_encoder2=nn.Linear(128,self.relation_classes+1)
        # self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST
        # self.relation_heads = build_relation_heads(cfg)
        # self.relation_classes = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        # self.positive_crowd_threshold = cfg.MODEL.ROI_HEADS.POSITIVE_CROWD_THRESHOLD
        # self.positive_box_threshold = cfg.MODEL.ROI_HEADS.POSITIVE_BOX_THRESHOLD

        # self.gt_similarity_mode = cfg.MODEL.ROI_HEADS.GT_SIMILARITY_MODE
        # self.similarity_mode = cfg.MODEL.ROI_HEADS.SIMILARITY_MODE
        # self.graph_search_mode = cfg.MODEL.ROI_HEADS.GRAPH_SEARCH_MODE
        #
        # self.similarity_loss = cfg.MODEL.ROI_HEADS.SIMILARITY_LOSS
        # self.train_only_same_class = cfg.MODEL.ROI_HEADS.SIMILARITY_TRAIN_ONLY_SAME_CLASS

        self.to(self.device)

    def forward(self, batched_inputs,iteration, mode, training, print_size=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        self.print_size=print_size
        losses = {}
        results=[]
        metrics={}

        images = self.preprocess_image(batched_inputs)

        output, features, pos = self.detr(images)
        src, mask = features[-1].decompose()
        bs, c, h, w = src.shape
        if self.mask_on:
            g_output,memory = self.gdetr(features, pos)
        else:
            g_output = self.gdetr(features, pos)

        if mode=="instance":
            if not training:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else None
                box_feature = output["pred_features"]
                instance_detections,keeps = self.inference(box_cls, box_pred, mask_pred, box_feature, images.image_sizes)
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes,
                                                     "instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(instance_results[i])
            else:
                print("not support instance training")
                exit()
        elif mode=="instance_mask_crowd":
            if training:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances)
                loss_dict = self.criterion(output, targets,trainable="masks")
                weight_dict = self.criterion.weight_dict
                for k in loss_dict.keys():
                    if k in weight_dict:
                        loss_dict[k] *= weight_dict[k]
                losses.update(loss_dict)

                gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
                # print(gt_instances[0])
                # print(gt_crowds[0])
                g_targets = self.prepare_targets(gt_crowds)
                g_loss_dict = self.criterion(g_output, g_targets)
                weight_dict = self.criterion.weight_dict
                key_list=list(g_loss_dict.keys())
                for k in key_list:
                    if k in weight_dict:
                        g_loss_dict["g_"+k] = g_loss_dict[k]*weight_dict[k]
                    del g_loss_dict[k]
                losses.update(g_loss_dict)
            else:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else None
                box_feature = output["pred_features"]
                instance_detections,keeps = self.inference(box_cls, box_pred, mask_pred, box_feature, images.image_sizes)
                print(instance_detections[0])
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes, "instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(instance_results[i])

                g_box_cls = g_output["pred_logits"]
                g_box_pred = g_output["pred_boxes"]
                g_mask_pred = g_output["pred_masks"] if self.mask_on else None
                g_box_feature = g_output["pred_features"]
                g_instance_detections,g_keeps = self.inference(g_box_cls, g_box_pred, g_mask_pred, g_box_feature, images.image_sizes)
                print(g_instance_detections[0])
                g_instance_results = self._postprocess(g_instance_detections, batched_inputs, images.image_sizes, "crowd")
                for i, r in enumerate(g_instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(g_instance_results[i])
        elif mode=="crowd":
            if training:
                gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
                g_targets = self.prepare_targets(gt_crowds,images.image_sizes)
                g_loss_dict = self.criterion(g_output, g_targets)
                weight_dict = self.criterion.weight_dict
                key_list=list(g_loss_dict.keys())
                for k in key_list:
                    g_loss_dict["g_" + k] = g_loss_dict[k]
                    if k in weight_dict:
                        g_loss_dict["g_"+k] *= weight_dict[k]
                    del g_loss_dict[k]
                losses.update(g_loss_dict)

                instance_detections,keeps = self.inference(output["pred_logits"], output["pred_boxes"], output["pred_masks"], output["pred_features"], images.image_sizes)
                g_instance_detections,g_keeps = self.inference(g_output["pred_logits"], g_output["pred_boxes"], g_output["pred_masks"], g_output["pred_features"], images.image_sizes)
                print([len(instance_detection) for instance_detection in instance_detections],
                      [len(g_instance_detection) for g_instance_detection in g_instance_detections])
                # instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes, "instance")
                # for i, r in enumerate(instance_results):
                #     if i >= len(results):
                #         results.append({})
                #     results[i].update(instance_results[i])
                # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                # for i, r in enumerate(gt_instances):
                #     gt_instances[i].pred_boxes = gt_instances[i].gt_boxes
                #     gt_instances[i].pred_classes = gt_instances[i].gt_classes
                #     gt_instances[i].pred_masks = F.interpolate(gt_instances[i].gt_masks.tensor.unsqueeze(0).float(), size=(batched_inputs[i]['height'],batched_inputs[i]['width']), mode='bilinear', align_corners=False)[0]>0.5
                #     results[i]['gt_instances'] = gt_instances[i]
                #
                # g_instance_results = self._postprocess(g_instance_detections, batched_inputs, images.image_sizes, "crowd")
                # for i, r in enumerate(g_instance_results):
                #     if i >= len(results):
                #         results.append({})
                #     results[i].update(g_instance_results[i])
                # for i, r in enumerate(gt_crowds):
                #     gt_crowds[i].pred_boxes=gt_crowds[i].gt_boxes
                #     gt_crowds[i].pred_classes = gt_crowds[i].gt_classes
                #     gt_crowds[i].pred_masks = F.interpolate(gt_crowds[i].gt_masks.tensor.unsqueeze(0).float(), size=(batched_inputs[i]['height'],batched_inputs[i]['width']), mode='bilinear', align_corners=False)[0]>0.5
                #     results[i]['gt_crowds']=gt_crowds[i]
            else:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else None
                box_feature = output["pred_features"]
                instance_detections,keeps = self.inference(box_cls, box_pred, mask_pred, box_feature, images.image_sizes)
                print(instance_detections[0])
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes, "instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(instance_results[i])

                g_box_cls = g_output["pred_logits"]
                g_box_pred = g_output["pred_boxes"]
                g_mask_pred = g_output["pred_masks"] if self.mask_on else None
                g_box_feature = g_output["pred_features"]
                g_instance_detections,g_keeps = self.inference(g_box_cls, g_box_pred, g_mask_pred, g_box_feature, images.image_sizes)
                print(g_instance_detections[0])
                g_instance_results = self._postprocess(g_instance_detections, batched_inputs, images.image_sizes, "crowd")
                for i, r in enumerate(g_instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(g_instance_results[i])
        elif mode=="crowd_relation":
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            box_feature = output["pred_features"]
            instance_detections, keeps = self.inference(box_cls, box_pred, mask_pred, box_feature, images.image_sizes, self.instance_threshold)
            # self.fullize_mask(instance_detections)
            instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes, "instance")
            for i, r in enumerate(instance_results):
                if i >= len(results):
                    results.append({})
                results[i].update(instance_results[i])

            g_box_cls = g_output["pred_logits"]
            g_box_pred = g_output["pred_boxes"]
            g_mask_pred = g_output["pred_masks"] if self.mask_on else None
            g_box_feature = g_output["pred_features"][-1]
            g_mask_feature = g_output["mask_features"]
            g_instance_detections, g_keeps = self.inference(g_box_cls, g_box_pred, g_mask_pred, g_box_feature, images.image_sizes, self.instance_threshold)
            # self.fullize_mask(g_instance_detections)
            # print(g_instance_detections[0])
            g_instance_results = self._postprocess(g_instance_detections, batched_inputs, images.image_sizes, "crowd")
            for i, r in enumerate(g_instance_results):
                if i >= len(results):
                    results.append({})
                results[i].update(g_instance_results[i])

            g_mask_pred = F.sigmoid(g_mask_pred)
            print([len(instance_detection) for instance_detection in instance_detections],
                  [len(g_instance_detection) for g_instance_detection in g_instance_detections])

            if training:
                gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
                gt_phrases = [x["phrases"].to(self.device) for x in batched_inputs]
                # print(gt_instances[0])
                # print(gt_crowds[0])
                g_targets = self.prepare_targets(gt_crowds,images.image_sizes)
                g_loss_dict = self.criterion(g_output, g_targets)
                weight_dict = self.criterion.weight_dict
                key_list=list(g_loss_dict.keys())
                for k in key_list:
                    g_loss_dict["g_" + k] = g_loss_dict[k]
                    if k in weight_dict:
                        g_loss_dict["g_"+k] *= weight_dict[k]
                    del g_loss_dict[k]
                losses.update(g_loss_dict)

                g_mask_gts=[]
                for g_mask_pred_single,gt_crowd in zip(g_mask_pred,gt_crowds):
                    pred_mask=g_mask_pred_single
                    gt_mask=gt_crowd.gt_masks.tensor
                    gt_mask_reshape=F.interpolate(gt_mask.unsqueeze(0).float(), size=pred_mask.shape[1:], mode='bilinear', align_corners=False)
                    g_mask_gts.append(gt_mask_reshape[0])

                # print("g_box_feature",g_box_feature.shape)
                # print("g_mask_feature",g_mask_feature.shape)
                pos_embed = pos[-1].unsqueeze(1).repeat(1,int(self.num_queries/5),1,1,1)
                query_embed = self.crowd_query_embed.weight.unsqueeze(1).repeat(1, bs*int(self.num_queries/5), 1)
                tgt = torch.zeros_like(query_embed)
                # print("tgt",tgt.shape)
                # print("memory",g_mask_feature.permute(3,4,0,1,2).flatten(0,1).flatten(1,2).shape)
                # print("mask",mask.unsqueeze(1).repeat(1,int(self.num_queries/5),1,1).flatten(0,1).flatten(1,2).shape)
                pos_embed = pos_embed.flatten(0, 1).flatten(2).permute(2, 0, 1)
                pos_embed_encoded=self.pos_encoder(pos_embed)
                # print("pos_embed", pos_embed_encoded.shape)
                # print("query_embed", query_embed.shape)
                hs = self.crowd_mask_decoder(tgt,
                                             g_mask_feature.permute(3,4,0,1,2).flatten(0,1).flatten(1,2), #1,20,8,10,36 --> 10,36,1,20,8 --> 360,20,8
                                             memory_key_padding_mask=mask.unsqueeze(1).repeat(1,int(self.num_queries/5),1,1).flatten(0,1).flatten(1,2), #1,10,36 --> 1,20,10,36 -> 20,360
                                             pos=pos_embed_encoded,
                                             query_pos=query_embed)
                # print("hs",hs.shape)
                g_mask_feature=hs[-1].squeeze(1).view(bs,int(self.num_queries/5),hs.shape[-1])
                # print("g_mask_feature",g_mask_feature.shape)
                candidate_subject_crowd_indexs = torch.range(0,g_box_feature.shape[-2]-1).unsqueeze(1).repeat(1,g_box_feature.shape[-2]).flatten().long().to(self.device)
                candidate_object_crowd_indexs = torch.range(0,g_box_feature.shape[-2]-1).unsqueeze(0).repeat(g_box_feature.shape[-2],1).flatten().long().to(self.device)
                positive_subject_crowd_indexs, positive_object_crowd_indexs, positive_relation_onehots, \
                negative_subject_crowd_indexs, negative_object_crowd_indexs, negative_relation_onehots\
                    =self.get_target_from_crowd_pred(instance_detections,g_mask_pred,g_box_cls,
                                        candidate_subject_crowd_indexs,candidate_object_crowd_indexs,
                                        g_mask_gts,gt_phrases)
                batch_i=0
                batch_length=[]
                subject_crowd_features=[]
                object_crowd_features = []
                gt_scores=[]
                for positive_subject_crowd_index, positive_object_crowd_index, positive_relation_onehot, \
                negative_subject_crowd_index, negative_object_crowd_index, negative_relation_onehot in zip(
                    positive_subject_crowd_indexs, positive_object_crowd_indexs, positive_relation_onehots,
                    negative_subject_crowd_indexs, negative_object_crowd_indexs, negative_relation_onehots):
                    batch_length.append(len(positive_subject_crowd_indexs))
                    positive_subject_crowd_feature = torch.cat([g_box_feature[batch_i, positive_subject_crowd_index],
                                                        g_mask_feature[batch_i, positive_subject_crowd_index]], dim=1)
                    positive_object_crowd_feature = torch.cat([g_box_feature[batch_i, positive_object_crowd_index],
                                                               g_mask_feature[batch_i, positive_object_crowd_index]],dim=1)
                    negative_subject_crowd_feature = torch.cat([g_box_feature[batch_i, negative_subject_crowd_index],
                                                        g_mask_feature[batch_i, negative_subject_crowd_index]], dim=1)
                    negative_object_crowd_feature = torch.cat([g_box_feature[batch_i, negative_object_crowd_index],
                                                               g_mask_feature[batch_i, negative_object_crowd_index]],dim=1)
                    subject_crowd_features.append(positive_subject_crowd_feature)
                    subject_crowd_features.append(negative_subject_crowd_feature)
                    object_crowd_features.append(positive_object_crowd_feature)
                    object_crowd_features.append(negative_object_crowd_feature)
                    gt_scores.append(positive_relation_onehot)
                    gt_scores.append(negative_relation_onehot)
                    batch_i += 1
                subject_crowd_features = torch.cat(subject_crowd_features)
                object_crowd_features = torch.cat(object_crowd_features)
                gt_scores=torch.cat(gt_scores)
                # print(subject_crowd_features.shape)
                # print(object_crowd_features.shape)
                # print(gt_scores.shape)
                pred_scores = F.sigmoid(self.crowd_encoder2(F.relu(self.crowd_encoder(subject_crowd_features - object_crowd_features))))
                loss_rel_pos, loss_rel_neg = self.binary_focal_loss(pred_scores.flatten(), gt_scores.flatten())
                losses['loss_rel_pos'], losses['loss_rel_neg']=10*loss_rel_pos,10*loss_rel_neg
            else:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else None
                box_feature = output["pred_features"]
                instance_detections,keeps = self.inference(box_cls, box_pred, mask_pred, box_feature, images.image_sizes)
                print(instance_detections[0])
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes, "instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(instance_results[i])

                g_box_cls = g_output["pred_logits"]
                g_box_pred = g_output["pred_boxes"]
                g_mask_pred = g_output["pred_masks"] if self.mask_on else None
                g_box_feature = g_output["pred_features"]
                g_instance_detections,g_keeps = self.inference(g_box_cls, g_box_pred, g_mask_pred, g_box_feature, images.image_sizes)
                print(g_instance_detections[0])
                g_instance_results = self._postprocess(g_instance_detections, batched_inputs, images.image_sizes, "crowd")
                for i, r in enumerate(g_instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(g_instance_results[i])
        elif mode=="relation":
            if training:
                pass
            else:
                pass
        return results, losses, metrics

    def prepare_targets(self, targets,image_sizes):
        new_targets = []
        for targets_per_image,image_size in zip(targets,image_sizes):
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                # gt_masks = F.interpolate(targets_per_image.gt_masks.tensor.unsqueeze(0).float(), size=image_size, mode='bilinear', align_corners=False)[0]>0.5
                # gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({'masks': targets_per_image.gt_masks.tensor})
        return new_targets

    def fullize_mask(self, instances):
        for instance_per_image in instances:
            height, width = instance_per_image.image_size
            fullize_mask(instance_per_image, height, width)

    def get_target_from_crowd_pred(self,instance_detections,crowd_mask_preds,crowd_class_preds,
                      candidate_subject_crowd_indexs, candidate_object_crowd_indexs,
                      crowd_mask_gts,gt_phrases):
        positive_subject_crowd_indexs=[]
        positive_object_crowd_indexs=[]
        positive_relation_onehots=[]
        negative_subject_crowd_indexs=[]
        negative_object_crowd_indexs=[]
        negative_relation_onehots=[]
        for instance_detection,crowd_mask_pred, crowd_class_pred, crowd_mask_gt, gt_phrase in zip(instance_detections,crowd_mask_preds,crowd_class_preds,crowd_mask_gts,gt_phrases):
            # print(instance_detection.pred_full_masks.shape)
            crowd_mask_pred=crowd_mask_pred#>0.5
            subject_crowd_masks=crowd_mask_pred[candidate_subject_crowd_indexs]
            object_crowd_masks=crowd_mask_pred[candidate_object_crowd_indexs]
            crowd_mask_gt=crowd_mask_gt>0.5
            # print(torch.sum(crowd_mask_pred,dim=[1,2]))
            # print(torch.sum(crowd_mask_gt,dim=[1,2]))
            subject_crowd_classes=crowd_class_pred[candidate_subject_crowd_indexs]
            object_crowd_classes=crowd_class_pred[candidate_object_crowd_indexs]

            extend_subject_classes = gt_phrase.gt_subject_classes.unsqueeze(0).repeat(len(candidate_subject_crowd_indexs),1)
            extend_subject_where=torch.where(extend_subject_classes>=0)
            subject_interaction_area=torch.sum(subject_crowd_masks.unsqueeze(1).repeat(1,len(gt_phrase),1,1)
                                                     *crowd_mask_gt[gt_phrase.gt_subject_crowd_ids].unsqueeze(0).repeat(len(candidate_subject_crowd_indexs),1,1,1),dim=[2,3])
            subject_union_area=torch.sum(subject_crowd_masks.unsqueeze(1).repeat(1,len(gt_phrase),1,1)
                                                     +crowd_mask_gt[gt_phrase.gt_subject_crowd_ids].unsqueeze(0).repeat(len(candidate_subject_crowd_indexs),1,1,1),dim=[2,3])
            subject_crowd_gt_phrase_paired=subject_interaction_area/subject_union_area>0.5\
                                           *subject_crowd_masks.shape[-1]*subject_crowd_masks.shape[-2] * \
                                           (subject_crowd_classes.unsqueeze(1).repeat(1, len(gt_phrase), 1)[(extend_subject_where[0], extend_subject_where[1],extend_subject_classes.flatten())] > 0.5).view(extend_subject_classes.shape[0], extend_subject_classes.shape[1])
            extend_object_classes=gt_phrase.gt_object_classes.unsqueeze(0).repeat(len(candidate_object_crowd_indexs), 1)
            extend_object_where=torch.where(extend_object_classes>=0)
            object_interaction_area=torch.sum(object_crowd_masks.unsqueeze(1).repeat(1, len(gt_phrase), 1, 1)
                                                    *crowd_mask_gt[gt_phrase.gt_object_crowd_ids].unsqueeze(0).repeat(len(candidate_object_crowd_indexs),1,1,1),dim=[2,3])
            object_union_area=torch.sum(object_crowd_masks.unsqueeze(1).repeat(1, len(gt_phrase), 1, 1)
                                                    +crowd_mask_gt[gt_phrase.gt_object_crowd_ids].unsqueeze(0).repeat(len(candidate_object_crowd_indexs),1,1,1),dim=[2,3])
            object_crowd_gt_phrase_paired=object_interaction_area/object_union_area>0.5\
                                          *subject_crowd_masks.shape[-1]*subject_crowd_masks.shape[-2] *\
                                          (object_crowd_classes.unsqueeze(1).repeat(1,len(gt_phrase),1)[(extend_object_where[0],extend_object_where[1],extend_object_classes.flatten())]>0.5).view(extend_object_classes.shape[0], extend_object_classes.shape[1])
            # print(subject_crowd_gt_phrase_paired.shape)
            # print(object_crowd_gt_phrase_paired.shape)
            matched_candidate_crowd_idx,matched_gt_phrase_idx=torch.where(subject_crowd_gt_phrase_paired*object_crowd_gt_phrase_paired)
            not_matched_candidate_crowd_idx, not_matched_gt_phrase_idx = torch.where(subject_crowd_gt_phrase_paired * object_crowd_gt_phrase_paired==0)
            randindx=torch.LongTensor(np.random.randint(0,len(not_matched_candidate_crowd_idx),size=min(max(5,len(matched_gt_phrase_idx)),len(not_matched_candidate_crowd_idx))))
            not_matched_candidate_crowd_idx=not_matched_candidate_crowd_idx[randindx]

            # print(matched_gt_phrase_idx.shape)
            positive_subject_crowd_index=candidate_subject_crowd_indexs[matched_candidate_crowd_idx]
            positive_object_crowd_index=candidate_object_crowd_indexs[matched_candidate_crowd_idx]
            positive_relation_onehot=gt_phrase.gt_relation_onehots[matched_gt_phrase_idx]
            # print(positive_relation_onehot.shape)
            negative_subject_crowd_index = candidate_subject_crowd_indexs[not_matched_candidate_crowd_idx]
            negative_object_crowd_index = candidate_object_crowd_indexs[not_matched_candidate_crowd_idx]
            negative_relation_onehot = torch.cat([torch.ones_like(negative_object_crowd_index).unsqueeze(1),
                                                   torch.zeros_like(gt_phrase.gt_relation_onehots[0][1:]).unsqueeze(0).repeat(len(negative_object_crowd_index),1).to(self.device)],dim=1)
            print("pos",len(positive_relation_onehot),"neg",len(negative_relation_onehot))
            positive_subject_crowd_indexs.append(positive_subject_crowd_index)
            positive_object_crowd_indexs.append(positive_object_crowd_index)
            positive_relation_onehots.append(positive_relation_onehot)
            negative_subject_crowd_indexs.append(negative_subject_crowd_index)
            negative_object_crowd_indexs.append(negative_object_crowd_index)
            negative_relation_onehots.append(negative_relation_onehot)
        return positive_subject_crowd_indexs, positive_object_crowd_indexs, positive_relation_onehots, \
                negative_subject_crowd_indexs, negative_object_crowd_indexs, negative_relation_onehots
    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes, mode="instance"):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({mode + "s": r})
        return processed_results

    def inference(self, box_cls, box_pred, mask_pred, box_feature, image_sizes, instance_threshold=0.5):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        keeps = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, box_feature_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, box_feature, image_sizes
        )):
            keep=scores_per_image>instance_threshold
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            # result.pred_features = box_feature_per_image
            result=result[keep]
            keeps.append(keep)
            results.append(result)
        return results,keeps

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def binary_focal_loss(self, pred, gt, pos_gamma=1.0, neg_gamma=2.0):
        # print("======================================")
        num_1 = torch.sum(gt).item() * 1.0
        num_0 = gt.shape[0] - num_1
        alpha = 0.5  # 1.0-num_1/gt.shape[0]
        # print(alpha)
        # print(pred)
        # print(gt)
        epsilon = 1.e-5
        pred = pred.clamp(epsilon, 1 - epsilon)
        ce_1 = gt * (-torch.log(pred))  # gt=1
        ce_0 = (1 - gt) * (-torch.log(1 - pred))  # gt=0

        # ce=ce_1+ce_0
        # ce_avg=torch.mean(ce)
        # print(ce_0.shape)
        # print("ce_1")
        # print(ce_1)
        # print("ce_0")
        # print(ce_0)

        fl_1 = torch.pow(1 - pred, pos_gamma) * ce_1
        # print("fl_1")
        # print(fl_1)
        # fl_1 = alpha*fl_1
        # print(fl_1)

        fl_0 = torch.pow(pred, neg_gamma) * ce_0
        # print("fl_0")
        # print(fl_0)
        # fl_0 = (1-alpha)*fl_0
        # print(fl_0)

        if num_1 == 0:
            fl_1_avg = torch.sum(fl_1)
        else:
            fl_1_avg = torch.sum(fl_1) / num_1
        if num_0 == 0:
            fl_0_avg = torch.sum(fl_0)
        else:
            fl_0_avg = torch.sum(fl_0) / num_0
        # fl=fl_0+fl_1
        # fl_avg=torch.mean(fl)
        # print(fl_avg)
        # print(fl_1_avg)
        # print(fl_0_avg)
        # print("======================================")
        return fl_1_avg, fl_0_avg
