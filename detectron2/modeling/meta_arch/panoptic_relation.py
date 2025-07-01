# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from matplotlib import pyplot as plt
import numpy as np

import torch
from torch import nn
from torchvision.transforms import Resize
from detectron2.structures import ImageList, Boxes, Instances
from detectron2.utils.torch_utils import extract_bbox
from detectron2.data import MetadataCatalog

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess, sem_seg_postprocess
from ..middleprocessing import generate_thing_instances, generate_stuff_instances,\
    generate_instances,generate_gt_instances,generate_pair_instances,generate_mannual_relation,\
    map_gt_and_relations,generate_instances_interest,generate_pairs_interest
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from ..relation_heads import build_relation_heads, build_instance_encoder
from .build import META_ARCH_REGISTRY
from .semantic_seg import build_sem_seg_head

__all__ = ["PanopticRelation"]


@META_ARCH_REGISTRY.register()
class PanopticRelation(nn.Module):
    """
    Main class for Panoptic FPN architectures (see https://arxiv.org/abs/1901.02446).
    """

    def __init__(self, cfg):
        super().__init__()

        self.square_image_size=cfg.MODEL.RELATION_HEADS.IMAGE_SIZE

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.instance_loss_weight = cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED
        self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH
        self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT
        self.combine_instances_confidence_threshold = (
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH
        )

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.relation_heads = build_relation_heads(cfg)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.thing_id_map = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_contiguous_id_to_class_id")
        self.stuff_id_map = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("stuff_contiguous_id_to_class_id")

        if self.thing_id_map is None:
            self.thing_id_map = {i: i + 1 for i in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}
        if self.stuff_id_map is None:
            self.stuff_id_map = {i + 1: i + cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1 for i in range(cfg.MODEL.RELATION_HEADS.INSTANCE_NUM-cfg.MODEL.ROI_HEADS.NUM_CLASSES-1)}
            self.stuff_id_map[0]=0

        self.relation_num = cfg.MODEL.RELATION_HEADS.RELATION_NUM
        self.relation_head_list=cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST
        self.instance_encoder=build_instance_encoder(cfg)

        self.to(self.device)

    def forward(self, batched_inputs,iteration,mode="panoptic",training=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                  See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
        if mode=="panoptic":
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
            features = self.backbone(images.tensor)

            if "proposals" in batched_inputs[0]:
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            if self.proposal_generator:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            detector_results, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances
            )

            if self.training:
                losses = {}
                losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
                losses.update(proposal_losses)
                return losses

            processed_results = []
            for detector_result, input_per_image, image_size in zip(
                detector_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                detector_r = detector_postprocess(detector_result, height, width)

                processed_results.append({"instances": detector_r})

                if self.combine_on:
                    panoptic_r = combine_semantic_and_instance_outputs(
                        self.thing_id_map,
                        self.stuff_id_map,
                        detector_r,
                        self.combine_overlap_threshold,
                        self.combine_stuff_area_limit,
                        self.combine_instances_confidence_threshold,
                        mode=mode,
                        device=self.device
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r
            return processed_results
        elif mode=="relation":
            losses={}
            metrics={}
            # print("origin image: ("+str(batched_inputs[0]['height'])+", "+str(batched_inputs[0]['width'])+")")
            images = [x["image"].to(self.device) for x in batched_inputs]
            # print("input image: "+str(images[0].shape))
            images = [self.normalizer(x) for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
            features = self.backbone(images.tensor)

            image_features = self.roi_heads.generate_instance_box_features(features, [Boxes(torch.IntTensor([[0, 0, image_size[1], image_size[0]]]).to(self.device)) for image_size in images.image_sizes])
            image_features=[image_feature for image_feature in image_features]
            if training:
                gt_instances = [batched_input['instances'].to(self.device) for batched_input in batched_inputs]
                gt_triplets = [batched_input['triplets'].to(self.device) for batched_input in batched_inputs]

                gt_instances = generate_instances_interest(gt_instances, gt_triplets,self.relation_num)
                gt_instance_nums=[len(gt_instance) for gt_instance in gt_instances]
                gt_box_features_mix = self.roi_heads.generate_instance_box_features(features, [gt_instance.pred_boxes for gt_instance in gt_instances])
                gt_box_features = gt_box_features_mix.split(gt_instance_nums)
                # print(gt_instances)

            if "proposals" in batched_inputs[0]:
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            if "instances" in batched_inputs[0]:
                gt_t_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_thing_instances = batched_inputs[0]['instances'].to(self.device)
            else:
                gt_t_instances = None
                gt_thing_instances = None
            if self.proposal_generator:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_t_instances) # rpn
            detector_results, thing_box_features = self.roi_heads(images, features, proposals, gt_t_instances,mode="relation")
            pred_instances, keep_areas = generate_thing_instances(self.thing_id_map,detector_results)
            pred_instance_nums=[len(pred_instance) for pred_instance in pred_instances]
            pred_box_features = []
            for i in range(len(pred_instance_nums)):
                keep_area=keep_areas[i]
                thing_box_feature=thing_box_features[i]
                if len(keep_area)>0:
                    pred_box_feature = thing_box_feature[keep_area[0]]
                else:
                    pred_box_feature=torch.Tensor().to(self.device)
                pred_box_features.append(pred_box_feature)
            # =================pred_instances,pred_box_features

            pred_pair_instances = None
            pred_pair_box_features = None
            pred_pair_instance_nums=[]
            if "pair" in self.relation_head_list or "predicate" in self.relation_head_list:
                pred_pair_instances = generate_pair_instances(pred_instances)
                pred_pair_instance_nums = [len(pred_pair_instance) for pred_pair_instance in pred_pair_instances]
                pred_pair_box_features_mix = self.roi_heads.generate_instance_box_features(features, [pred_pair_instance.pred_pair_boxes for pred_pair_instance in pred_pair_instances])
                pred_pair_box_features = pred_pair_box_features_mix.split(pred_pair_instance_nums)

            pred_mannual_triplets = None
            if training:
                pred_instances, pred_gt_triplets = map_gt_and_relations(pred_instances, gt_instances, gt_triplets)
                pred_instances = generate_instances_interest(pred_instances, pred_gt_triplets,self.relation_num)
                if "pair" in self.relation_head_list or "predicate" in self.relation_head_list:
                    pred_pair_instances = generate_pairs_interest(pred_instances,pred_pair_instances, pred_gt_triplets, self.relation_num)
                if "predicate" in self.relation_head_list:
                    pred_mannual_triplets = None#generate_mannual_relation(pred_instances, pred_gt_triplets)

            pred_instance_features, pred_instance_logits, pred_class_loss, pred_class_metirc = self.instance_encoder(
                image_features, pred_instances, pred_box_features, training)
            if pred_instance_logits is not None:
                for i in range(len(pred_instance_nums)):
                    pred_instances[i].pred_class_logits = pred_instance_logits[i]
                    instance_num = len(pred_instances[i])
                    subject_logits = pred_instance_logits[i][:, 1:].repeat(instance_num, 1, 1).permute(1, 0,2).contiguous().view(instance_num * instance_num, -1)
                    object_logits = pred_instance_logits[i][:, 1:].repeat(instance_num, 1, 1).contiguous().view(instance_num * instance_num, -1)
                    pred_pair_instances[i].pred_subject_logits=subject_logits
                    pred_pair_instances[i].pred_object_logits = object_logits

            if training:
                losses.update(pred_class_loss)
                metrics.update(pred_class_metirc)

            pred_pair_instance_features=None
            pred_pair_predicate_features = None
            if "pair" in self.relation_head_list or "predicate" in self.relation_head_list:
                pred_pair_instance_features, _, _, _ = self.instance_encoder(
                    image_features, pred_pair_instances, pred_pair_box_features, training=False)

                pred_pair_predicate_features = []
                for k in range(len(pred_instance_nums)):
                    pred_instance=pred_instances[k]
                    pred_instance_feature=pred_instance_features[k]
                    pred_pair_instance_feature=pred_pair_instance_features[k]

                    pred_pair_predicate_feature = []
                    for i in range(len(pred_instance)):
                        for j in range(len(pred_instance)):
                            pred_pair_predicate_feature.append(torch.cat([pred_instance_feature[i] - pred_instance_feature[j],pred_pair_instance_feature[i * pred_instance_feature.shape[0] + j]]))
                    pred_pair_predicate_feature = torch.stack(pred_pair_predicate_feature)
                    pred_pair_predicate_features.append(pred_pair_predicate_feature)
            # ==================pred_instances,pred_pair_instances,pred_pair_box_features,pred_pair_predicate_features


            relation_results, relation_losses, relation_metrics = self.relation_heads(image_features,pred_instances, pred_pair_instances,
                                                                                      pred_instance_features, pred_pair_instance_features, pred_pair_predicate_features,
                                                                                      pred_mannual_triplets,training,iteration)
            losses.update(relation_losses)
            metrics.update(relation_metrics)
            for name in losses:
                losses[name]=losses[name]*1.0/len(batched_inputs)
            return pred_instances, relation_results, losses, metrics

def combine_semantic_and_instance_outputs(
        thing_id_map,
        stuff_id_map,
    instance_results,
    overlap_threshold,
    stuff_area_limit,
    instances_confidence_threshold,
    mode="panoptic",
    device="cuda"
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0
    segments_info = []

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_confidence_threshold:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        box=extract_bbox(mask).to(device)

        current_segment_id += 1

        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": instance_results.pred_classes[inst_id].item(),
                "class_id": thing_id_map[instance_results.pred_classes[inst_id].item()],
                "instance_id": inst_id.item(),
                "mask": mask,
                "box": box
            }
        )

    return segments_info
