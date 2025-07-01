# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss
import torchvision
from collections import defaultdict

# from detr.transformer import TransformerEncoderLayer, TransformerEncoder
from detectron2.layers import ShapeSpec
from detectron2.structures import ImageList, Instances, Boxes, BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.utils.torch_utils import box_iou, mask_iou, compute_boxes_io, compute_masks_io, union_boxes, union_box_list, union_box_matrix, SelfGCNLayer, DEC, calc_pairwise_distance, calc_self_distance, GAT, GATlast, GAT_su, GAT_fc, GCN
from ..backbone import build_backbone
from ..postprocessing import detector_postprocess, fullize_mask
from ..proposal_generator import build_proposal_generator, build_crowd_proposal_generator, build_phrase_proposal_generator
from ..roi_heads import build_roi_heads, build_phrase_roi_heads, build_crowd_roi_heads
from ..relation_heads import build_relation_heads
from .build import META_ARCH_REGISTRY
from ..roi_heads.box_head import build_box_head

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        self.print_size=False
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.similarity = cfg.SIMILARITY
        self.use_gt_instance=cfg.MODEL.RPN.USE_GT_INSTANCE
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        self.roi_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST
        self.relation_heads = build_relation_heads(cfg)
        self.relation_classes = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        self.positive_crowd_threshold = cfg.MODEL.ROI_HEADS.POSITIVE_CROWD_THRESHOLD
        # self.negative_crowd_threshold = cfg.MODEL.ROI_HEADS.NEGATIVE_CROWD_THRESHOLD
        self.positive_box_threshold = cfg.MODEL.ROI_HEADS.POSITIVE_BOX_THRESHOLD
        self.which_gt_matched = cfg.MODEL.ROI_HEADS.WHICH_GT_MATCHED

        self.gt_similarity_mode = cfg.MODEL.ROI_HEADS.GT_SIMILARITY_MODE
        self.similarity_mode = cfg.MODEL.ROI_HEADS.SIMILARITY_MODE
        self.graph_search_mode = cfg.MODEL.ROI_HEADS.GRAPH_SEARCH_MODE

        self.similarity_loss = cfg.MODEL.ROI_HEADS.SIMILARITY_LOSS
        self.train_only_same_class = cfg.MODEL.ROI_HEADS.SIMILARITY_TRAIN_ONLY_SAME_CLASS

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.instance_encoder = None
        if cfg.MODEL.ROI_HEADS.ENCODE_DIM>0:
            self.instance_encoder = nn.Linear(cfg.MODEL.ROI_HEADS.ENCODE_DIM,cfg.MODEL.ROI_HEADS.ENCODE_DIM_OUT)

        self.to(self.device)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = " 1. GT bounding boxes  2. Predicted proposals"
            storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs,iteration, mode, training,print_size=False):
        self.print_size=print_size
        losses = {}
        results=[]
        metrics={}

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if mode=="instance":
            if training:
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                elif "targets" in batched_inputs[0]:
                    log_first_n(
                        logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
                    )
                    gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
                else:
                    gt_instances = None

                if self.proposal_generator:
                    instance_proposals, instance_proposal_losses,  instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, gt_instances,training=True)
                else:
                    assert "proposals" in batched_inputs[0]
                    instance_proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                    instance_proposal_losses = {}

                instance_detections, instance_features, instance_detector_losses = self.roi_heads(images, features, instance_proposals, gt_instances,"instance",training=True)
                if instance_detections[0].has("proposal_boxes"):
                    for image_i in range(len(instance_detections)):
                        instance_detections[image_i].pred_boxes = instance_detections[image_i].proposal_boxes
                if self.vis_period > 0:
                    storage = get_event_storage()
                    if storage.iter % self.vis_period == 0:
                        self.visualize_training(batched_inputs, instance_proposals)

                instance_results = self._postprocess(instance_detections, batched_inputs,images.image_sizes, "instance")
                for i,r in enumerate(instance_results):
                    if i>=len(results):
                        results.append({})
                    results[i].update(instance_results[i])
                losses.update(instance_proposal_losses)
                losses.update(instance_detector_losses)
            else:
                if "proposals" in batched_inputs[0]:
                    instance_proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                    features_list = [features[f] for f in self.roi_in_features]
                    instance_detections = self.roi_heads._forward_mask(features_list,instance_proposals,training=False)
                else:
                    instance_proposals, _,  instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                    instance_detections, instance_features, instance_detector_loss = self.roi_heads(images, features, instance_proposals, None,"instance",False)

                if self.print_size:
                    print(len(instance_detections[0]))
                # instances
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes,"instance")

                for i,r in enumerate(instance_results):
                    if i>=len(results):
                        results.append({})
                    results[i].update(instance_results[i])

        return results, losses, metrics

    def single_area_generate_to_pred_phrases_from_to_pred_crowds(self,pred_instance,to_pred_crowd,gt_crowd,gt_phrase,training=True):
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
            to_pred_phrase.pred_subject_crowd_ids = to_pred_phrase_pred_subject_crowd_ids.long()
            to_pred_phrase.pred_object_crowd_ids = to_pred_phrase_pred_object_crowd_ids.long()
            to_pred_phrase.pred_subject_lens = to_pred_crowd.pred_instance_lens[to_pred_phrase_pred_subject_crowd_ids].long()
            to_pred_phrase.pred_object_lens = to_pred_crowd.pred_instance_lens[to_pred_phrase_pred_object_crowd_ids].long()
            to_pred_phrase.pred_subject_boxes = Boxes(to_pred_crowd.pred_boxes.tensor[to_pred_phrase_pred_subject_crowd_ids])
            to_pred_phrase.pred_object_boxes = Boxes(to_pred_crowd.pred_boxes.tensor[to_pred_phrase_pred_object_crowd_ids])
            to_pred_phrase.pred_subject_in_clust  = to_pred_crowd.pred_in_clust[to_pred_phrase_pred_subject_crowd_ids]
            to_pred_phrase.pred_object_in_clust = to_pred_crowd.pred_in_clust[to_pred_phrase_pred_object_crowd_ids]
            to_pred_phrase.pred_boxes = Boxes(union_boxes(to_pred_phrase.pred_subject_boxes.tensor, to_pred_phrase.pred_object_boxes.tensor))
            filter_single = torch.where((to_pred_crowd.pred_instance_lens[to_pred_phrase_pred_subject_crowd_ids] > 1) + (to_pred_crowd.pred_instance_lens[to_pred_phrase_pred_object_crowd_ids] > 1) > 0)
            to_pred_phrase=to_pred_phrase[filter_single]

            if training:
                to_pred_phrase = self.single_area_choose_phrase_for_training(pred_instance,to_pred_phrase,gt_phrase)
        else:
            to_pred_phrase.pred_subject_crowd_ids=torch.Tensor().to(self.device)
            to_pred_phrase.pred_object_crowd_ids=torch.Tensor().to(self.device)
            to_pred_phrase.pred_subject_classes=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_object_classes=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_subject_ids_list=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_object_ids_list=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_subject_lens=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_object_lens=torch.Tensor().to(self.device).long()
            to_pred_phrase.pred_subject_boxes=torch.Tensor().to(self.device)
            to_pred_phrase.pred_object_boxes=torch.Tensor().to(self.device)
            to_pred_phrase.pred_boxes=Boxes(torch.Tensor().to(self.device))
            to_pred_phrase.pred_subject_in_clust = torch.Tensor().to(self.device)
            to_pred_phrase.pred_object_in_clust = torch.Tensor().to(self.device)
            if training:
                to_pred_phrase.pred_subject_gt_in_crowds_list = torch.Tensor().to(self.device)
                to_pred_phrase.pred_object_gt_in_crowds_list = torch.Tensor().to(self.device)
                to_pred_phrase.gt_relation_onehots = torch.Tensor().to(self.device)
                # to_pred_phrase.to_pred_phrase_subobj_crowd_pred_mask_ios_gt_mask_topk = torch.Tensor().to(self.device)
        return to_pred_crowd, to_pred_phrase

    def single_area_choose_phrase_for_training(self, pred_instance, to_pred_phrase, gt_phrase):
        pred_instance_boxes_pad0 = torch.cat([pred_instance.pred_boxes.tensor, torch.Tensor([[pred_instance.image_size[1], pred_instance.image_size[0], 0, 0]]).to(self.device)])

        pred_subject_ids = to_pred_phrase.pred_subject_ids_list.unsqueeze(1).repeat(1, len(gt_phrase), 1).unsqueeze(3).repeat(1, 1, 1, gt_phrase.gt_subject_ids_list.shape[1])  # phrase,gt_phrase,pred_instance,gt_instance
        pred_object_ids = to_pred_phrase.pred_object_ids_list.unsqueeze(1).repeat(1, len(gt_phrase), 1).unsqueeze(3).repeat(1, 1, 1, gt_phrase.gt_object_ids_list.shape[1])  # phrase,gt_phrase,pred_instance,gt_instance
        pred_subject_ids[pred_subject_ids == -1] = -2
        pred_object_ids[pred_object_ids == -1] = -2

        gt_subject_ids = gt_phrase.gt_subject_ids_list.unsqueeze(1).repeat(1, to_pred_phrase.pred_subject_ids_list.shape[1], 1).unsqueeze(0).repeat(len(to_pred_phrase), 1, 1, 1)  # phrase,gt_phrase,instance,gt_instance
        gt_object_ids = gt_phrase.gt_object_ids_list.unsqueeze(1).repeat(1, to_pred_phrase.pred_subject_ids_list.shape[1], 1).unsqueeze(0).repeat(len(to_pred_phrase), 1, 1, 1)  # phrase,gt_phrase,instance,gt_instance
        # print(pred_subject_ids.shape,gt_subject_ids.shape)
        to_pred_phrase_how_many_subject_in_gt = (torch.sum(pred_subject_ids == gt_subject_ids, dim=3) > 0).float()  # phrase,gt_phrase,instance
        to_pred_phrase_how_many_object_in_gt = (torch.sum(pred_object_ids == gt_object_ids, dim=3) > 0).float()  # phrase,gt_phrase,instance
        to_pred_phrase_how_many_gt_subject_matched = (torch.sum(pred_subject_ids == gt_subject_ids, dim=2) > 0).float()  # phrase,gt_phrase,gt_instance
        to_pred_phrase_how_many_gt_object_matched = (torch.sum(pred_object_ids == gt_object_ids, dim=2) > 0).float()  # phrase,gt_phrase,gt_instance

        to_pred_phrase_how_percent_subject_in_gt = (torch.sum(to_pred_phrase_how_many_subject_in_gt,dim=2).float() / to_pred_phrase.pred_subject_lens.unsqueeze(1).repeat(1,len(gt_phrase)))  # phrase, gt_phrase
        to_pred_phrase_how_percent_object_in_gt = (torch.sum(to_pred_phrase_how_many_object_in_gt, dim=2).float() / to_pred_phrase.pred_object_lens.unsqueeze(1).repeat(1,len(gt_phrase)))  # phrase, gt_phrase
        to_pred_phrase_how_percent_gt_subject_matched = (torch.sum(to_pred_phrase_how_many_gt_subject_matched, dim=2).float() / gt_phrase.gt_subject_lens.unsqueeze(0).repeat(len(to_pred_phrase),1)) # phrase, gt_phrase
        to_pred_phrase_how_percent_gt_object_matched = (torch.sum(to_pred_phrase_how_many_gt_object_matched,dim=2).float() / gt_phrase.gt_object_lens.unsqueeze(0).repeat(len(to_pred_phrase),1)) # phrase, gt_phrase

        if self.which_gt_matched=="all":
            positive_to_pred_phrase_idx_gt_phrase_idx = torch.where(((to_pred_phrase_how_percent_gt_subject_matched >= self.positive_crowd_threshold) *
                                                                     (to_pred_phrase_how_percent_gt_object_matched >= self.positive_crowd_threshold)) > 0)  # positive: phrase,gt_phrase
        elif self.which_gt_matched=="max":
            ## each candidate mapping to the best matching gt
            # to_pred_phrase, gt_phrase: how many percent instance in gt * to_pred_phrase, gt_phrase: if gt percent matched
            to_pred_phrase_matching_gt_value, to_pred_phrase_matching_gt_idx = torch.max((to_pred_phrase_how_percent_subject_in_gt * to_pred_phrase_how_percent_object_in_gt) * \
                ((to_pred_phrase_how_percent_gt_subject_matched >= self.positive_crowd_threshold) * (to_pred_phrase_how_percent_gt_object_matched >= self.positive_crowd_threshold)), dim=1)
            to_pred_phrase_idx = torch.arange(0, len(to_pred_phrase)).to(self.device)
            to_pred_phrase_idx = to_pred_phrase_idx[torch.where(to_pred_phrase_matching_gt_value > 0)]
            to_pred_phrase_matching_gt_idx = to_pred_phrase_matching_gt_idx[torch.where(to_pred_phrase_matching_gt_value > 0)]
            positive_to_pred_phrase_idx_gt_phrase_idx = (to_pred_phrase_idx, to_pred_phrase_matching_gt_idx)
        else:
            print("NO SUCH GT MATCHED:",self.which_gt_matched)
            exit()
        positive_to_pred_phrase_how_percent_gt_subject_matched = to_pred_phrase_how_percent_gt_subject_matched[positive_to_pred_phrase_idx_gt_phrase_idx]
        positive_to_pred_phrase_how_percent_gt_object_matched = to_pred_phrase_how_percent_gt_object_matched[positive_to_pred_phrase_idx_gt_phrase_idx]
        positive_topk = min(positive_to_pred_phrase_how_percent_gt_subject_matched.shape[0], 500)
        positive_to_pred_phrase_how_percent_subobj_gt_matched_topk, positive_to_pred_phrase_select_by_mask_topk = torch.topk(-positive_to_pred_phrase_how_percent_gt_subject_matched * positive_to_pred_phrase_how_percent_gt_object_matched,positive_topk)
        positive_to_pred_phrase = to_pred_phrase[positive_to_pred_phrase_idx_gt_phrase_idx[0]][positive_to_pred_phrase_select_by_mask_topk]
        # positive_to_pred_phrase.remove("pred_subject_crowd_ids")
        # positive_to_pred_phrase.remove("pred_object_crowd_ids")
        # print("to_pred_phrase sub ids",to_pred_phrase.pred_subject_ids_list)
        # print("to_pred_phrase obj ids", to_pred_phrase.pred_object_ids_list)
        # print("positive_to_pred_phrase_idx_gt_phrase_idx",positive_to_pred_phrase_idx_gt_phrase_idx)
        negative_to_pred_phrase_idx=[]
        for i in range(len(to_pred_phrase)):
            if i not in positive_to_pred_phrase_idx_gt_phrase_idx[0]:
                negative_to_pred_phrase_idx.append(i)
        # print("negative_to_pred_phrase_idx",negative_to_pred_phrase_idx)
        negative_to_pred_phrase_idx=torch.Tensor(negative_to_pred_phrase_idx).to(self.device).long()
        negative_to_pred_phrase_idx=negative_to_pred_phrase_idx[torch.randint(0, len(negative_to_pred_phrase_idx),
                                                                              [min(min(len(negative_to_pred_phrase_idx),len(positive_to_pred_phrase)),500-len(positive_to_pred_phrase))])]
        # print("negative_to_pred_phrase_idx",negative_to_pred_phrase_idx)
        negative_to_pred_phrase=to_pred_phrase[negative_to_pred_phrase_idx]

        positive_to_pred_phrase.gt_relation_onehots = gt_phrase.gt_relation_onehots[positive_to_pred_phrase_idx_gt_phrase_idx[1]][positive_to_pred_phrase_select_by_mask_topk]
        negative_to_pred_phrase.gt_relation_onehots = torch.cat([torch.ones((len(negative_to_pred_phrase), 1)),
                                                                 torch.zeros((len(negative_to_pred_phrase),self.relation_classes))],dim=1).to(self.device)

        positive_to_pred_phrase.pred_subject_gt_in_crowds_list=to_pred_phrase_how_many_subject_in_gt[positive_to_pred_phrase_idx_gt_phrase_idx][positive_to_pred_phrase_select_by_mask_topk]
        positive_to_pred_phrase.pred_object_gt_in_crowds_list =to_pred_phrase_how_many_object_in_gt[positive_to_pred_phrase_idx_gt_phrase_idx][positive_to_pred_phrase_select_by_mask_topk]
        positive_to_pred_phrase.confidence=torch.ones_like(positive_to_pred_phrase.pred_subject_classes)

        negative_subject_in_crowd = torch.zeros_like(negative_to_pred_phrase.pred_subject_ids_list)
        negative_subject_in_crowd[negative_to_pred_phrase.pred_subject_ids_list >= 0] = 1
        negative_object_in_crowd = torch.zeros_like(negative_to_pred_phrase.pred_object_ids_list)
        negative_object_in_crowd[negative_to_pred_phrase.pred_object_ids_list >= 0] = 1
        negative_to_pred_phrase.pred_subject_gt_in_crowds_list = negative_subject_in_crowd
        negative_to_pred_phrase.pred_object_gt_in_crowds_list = negative_object_in_crowd
        negative_to_pred_phrase.confidence = torch.zeros_like(negative_to_pred_phrase.pred_subject_classes)

        to_pred_phrase = Instances.cat([positive_to_pred_phrase, negative_to_pred_phrase])

        if len(to_pred_phrase)>0:
            to_pred_phrase = to_pred_phrase[torch.randperm(len(to_pred_phrase))]
            first = torch.where(to_pred_phrase.confidence == 1)[0][0:1]
            to_pred_phrase=Instances.cat([to_pred_phrase[:first],
                                          to_pred_phrase[first+1:],
                                          to_pred_phrase[first]])

        # to_pred_phrase.pred_subject_crowd_ids = torch.arange(0, len(to_pred_phrase)).to(self.device).long()
        # to_pred_phrase.pred_object_crowd_ids = torch.arange(len(to_pred_phrase), len(to_pred_phrase) * 2).to(self.device).long()

        # to_pred_crowd = Instances(to_pred_phrase.image_size)
        # to_pred_crowd.pred_boxes=Boxes.cat([to_pred_phrase.pred_subject_boxes,to_pred_phrase.pred_object_boxes])
        # to_pred_crowd.pred_instance_ids_list=torch.cat([to_pred_phrase.pred_subject_ids_list,to_pred_phrase.pred_object_ids_list])
        # to_pred_crowd.pred_in_clust = torch.cat([to_pred_phrase.pred_subject_in_clust,to_pred_phrase.pred_object_in_clust])
        # return to_pred_crowd
        if self.print_size:
            print("positive:", len(positive_to_pred_phrase), ", negative:", len(negative_to_pred_phrase))
            # print("---------------------------------------------")
            # print("instance in gt crowd \n", gt_crowd.gt_instance_ids_list)
            # print("instance in pred crowd \n", to_pred_crowd.pred_instance_ids_list)
            # print("---------------------------------------------")
            # print("positive gt sub \n", gt_phrase.gt_subject_ids_list)
            # print("positive gt obj \n", gt_phrase.gt_object_ids_list)
            # print("---------------------------------------------")
            # print("positive to pred phrase", len(positive_to_pred_phrase)) #,positive_to_pred_phrase_how_percent_subobj_gt_matched_topk.shape[0])
            # print("positive pred sub \n",positive_to_pred_phrase.pred_subject_ids_list)
            # print("positive pred obj \n",positive_to_pred_phrase.pred_object_ids_list)
            # print("---------------------------------------------")
            # print("negative to pred phrase", len(negative_to_pred_phrase))
            # print("negative pred sub \n", negative_to_pred_phrase.pred_subject_ids_list)
            # print("negative pred obj \n", negative_to_pred_phrase.pred_object_ids_list)
            # print("---------------------------------------------")
        return to_pred_phrase

    # ## old version
    # def single_area_choose_phrase_for_training(self, pred_instance, to_pred_phrase, gt_phrase):
    #     pred_instance_boxes_pad0 = torch.cat([pred_instance.pred_boxes.tensor, torch.Tensor([[pred_instance.image_size[1], pred_instance.image_size[0], 0, 0]]).to(self.device)])
    #
    #     pred_subject_ids = to_pred_phrase.pred_subject_ids_list.unsqueeze(1).repeat(1, len(gt_phrase), 1).unsqueeze(3).repeat(1, 1, 1, gt_phrase.gt_subject_ids_list.shape[1])  # phrase,gt_phrase,pred_instance,gt_instance
    #     pred_object_ids = to_pred_phrase.pred_object_ids_list.unsqueeze(1).repeat(1, len(gt_phrase), 1).unsqueeze(3).repeat(1, 1, 1, gt_phrase.gt_object_ids_list.shape[1])  # phrase,gt_phrase,pred_instance,gt_instance
    #     pred_subject_ids[pred_subject_ids == -1] = -2
    #     pred_object_ids[pred_object_ids == -1] = -2
    #
    #     gt_subject_ids = gt_phrase.gt_subject_ids_list.unsqueeze(1).repeat(1, to_pred_phrase.pred_subject_ids_list.shape[1], 1).unsqueeze(0).repeat(len(to_pred_phrase), 1, 1, 1)  # phrase,gt_phrase,instance,gt_instance
    #     gt_object_ids = gt_phrase.gt_object_ids_list.unsqueeze(1).repeat(1, to_pred_phrase.pred_subject_ids_list.shape[1], 1).unsqueeze(0).repeat(len(to_pred_phrase), 1, 1, 1)  # phrase,gt_phrase,instance,gt_instance
    #     # print(pred_subject_ids.shape,gt_subject_ids.shape)
    #     to_pred_phrase_how_many_subject_in_gt = (torch.sum(pred_subject_ids == gt_subject_ids, dim=3) > 0).float()  # phrase,gt_phrase,instance
    #     to_pred_phrase_how_many_object_in_gt = (torch.sum(pred_object_ids == gt_object_ids, dim=3) > 0).float()  # phrase,gt_phrase,instance
    #     to_pred_phrase_how_many_gt_subject_matched = (torch.sum(pred_subject_ids == gt_subject_ids, dim=2) > 0).float()  # phrase,gt_phrase,gt_instance
    #     to_pred_phrase_how_many_gt_object_matched = (torch.sum(pred_object_ids == gt_object_ids, dim=2) > 0).float()  # phrase,gt_phrase,gt_instance
    #
    #     to_pred_phrase_how_percent_subject_gt_matched = (torch.sum(to_pred_phrase_how_many_gt_subject_matched, dim=2).float() / gt_phrase.gt_subject_lens.unsqueeze(0).repeat(len(to_pred_phrase),1)) # phrase, gt_phrase
    #     to_pred_phrase_how_percent_object_gt_matched = (torch.sum(to_pred_phrase_how_many_gt_object_matched,dim=2).float() / gt_phrase.gt_object_lens.unsqueeze(0).repeat(len(to_pred_phrase),1)) # phrase, gt_phrase
    #
    #     positive_to_pred_phrase_idx_gt_phrase_idx = torch.where(((to_pred_phrase_how_percent_subject_gt_matched >= self.positive_crowd_threshold) *
    #                                                              (to_pred_phrase_how_percent_object_gt_matched >= self.positive_crowd_threshold)) > 0)  # positive: phrase,gt_phrase
    #     positive_to_pred_phrase_how_percent_subject_gt_matched = to_pred_phrase_how_percent_subject_gt_matched[positive_to_pred_phrase_idx_gt_phrase_idx]
    #     positive_to_pred_phrase_how_percent_object_gt_matched = to_pred_phrase_how_percent_object_gt_matched[positive_to_pred_phrase_idx_gt_phrase_idx]
    #     positive_topk = min(positive_to_pred_phrase_how_percent_subject_gt_matched.shape[0], 500)
    #     positive_to_pred_phrase_how_percent_subobj_gt_matched_topk, positive_to_pred_phrase_select_by_mask_topk = torch.topk(positive_to_pred_phrase_how_percent_subject_gt_matched + positive_to_pred_phrase_how_percent_object_gt_matched,positive_topk)
    #     positive_to_pred_phrase = to_pred_phrase[positive_to_pred_phrase_idx_gt_phrase_idx[0]][positive_to_pred_phrase_select_by_mask_topk]
    #     positive_to_pred_phrase.remove("pred_subject_crowd_ids")
    #     positive_to_pred_phrase.remove("pred_object_crowd_ids")
    #
    #     negative_to_pred_phrase_idx_gt_phrase_idx = torch.where(((to_pred_phrase_how_percent_subject_gt_matched < self.positive_crowd_threshold) +
    #                                                              (to_pred_phrase_how_percent_object_gt_matched < self.positive_crowd_threshold)) > 0)  # negative: phrase,gt_phrase
    #     negative_candidate_to_pred_phrase=to_pred_phrase[negative_to_pred_phrase_idx_gt_phrase_idx[0]]
    #     subobj_instance_ids_list=torch.stack([negative_candidate_to_pred_phrase.pred_subject_ids_list,negative_candidate_to_pred_phrase.pred_object_ids_list],dim=1)
    #     subobj_instance_ids_list=torch.unique(subobj_instance_ids_list,dim=0).long()
    #     negative_candidate_to_pred_phrase_pred_subject_ids_list = subobj_instance_ids_list[:,0,:]
    #     negative_candidate_to_pred_phrase_pred_object_ids_list = subobj_instance_ids_list[:,1,:]
    #     negative_to_pred_phrase=Instances(pred_instance.image_size)
    #     negative_to_pred_phrase.pred_subject_ids_list = negative_candidate_to_pred_phrase_pred_subject_ids_list
    #     negative_to_pred_phrase.pred_object_ids_list = negative_candidate_to_pred_phrase_pred_object_ids_list
    #     negative_to_pred_phrase.pred_subject_classes = pred_instance.pred_classes[negative_candidate_to_pred_phrase_pred_subject_ids_list[:,0]]
    #     negative_to_pred_phrase.pred_object_classes = pred_instance.pred_classes[negative_candidate_to_pred_phrase_pred_object_ids_list[:,0]]
    #     negative_to_pred_phrase.pred_subject_lens = torch.sum(negative_to_pred_phrase.pred_subject_ids_list>=0,dim=1).long()
    #     negative_to_pred_phrase.pred_object_lens = torch.sum(negative_to_pred_phrase.pred_object_ids_list>=0,dim=1).long()
    #     negative_to_pred_phrase.pred_subject_boxes = Boxes(union_box_matrix(pred_instance_boxes_pad0[negative_candidate_to_pred_phrase_pred_subject_ids_list]))
    #     negative_to_pred_phrase.pred_object_boxes = Boxes(union_box_matrix(pred_instance_boxes_pad0[negative_candidate_to_pred_phrase_pred_object_ids_list]))
    #     negative_to_pred_phrase.pred_boxes= Boxes(union_boxes(negative_to_pred_phrase.pred_subject_boxes.tensor,negative_to_pred_phrase.pred_object_boxes.tensor))
    #
    #     positive_to_pred_phrase.gt_relation_onehots = gt_phrase.gt_relation_onehots[positive_to_pred_phrase_idx_gt_phrase_idx[1]][positive_to_pred_phrase_select_by_mask_topk]
    #     negative_to_pred_phrase.gt_relation_onehots = torch.cat([torch.ones((len(negative_to_pred_phrase), 1)),
    #                                                              torch.zeros((len(negative_to_pred_phrase),self.relation_classes))],dim=1).to(self.device)
    #     positive_to_pred_phrase.pred_subject_gt_in_crowds_list=to_pred_phrase_how_many_subject_in_gt[positive_to_pred_phrase_idx_gt_phrase_idx][positive_to_pred_phrase_select_by_mask_topk]
    #     positive_to_pred_phrase.pred_object_gt_in_crowds_list =to_pred_phrase_how_many_object_in_gt[positive_to_pred_phrase_idx_gt_phrase_idx][positive_to_pred_phrase_select_by_mask_topk]
    #     positive_to_pred_phrase.confidence=torch.ones_like(positive_to_pred_phrase.pred_subject_classes)
    #
    #     negative_subject_in_crowd = torch.zeros_like(negative_to_pred_phrase.pred_subject_ids_list)
    #     negative_subject_in_crowd[negative_to_pred_phrase.pred_subject_ids_list >= 0] = 1
    #     negative_object_in_crowd = torch.zeros_like(negative_to_pred_phrase.pred_object_ids_list)
    #     negative_object_in_crowd[negative_to_pred_phrase.pred_object_ids_list >= 0] = 1
    #     negative_to_pred_phrase.pred_subject_gt_in_crowds_list = negative_subject_in_crowd
    #     negative_to_pred_phrase.pred_object_gt_in_crowds_list = negative_object_in_crowd
    #     negative_to_pred_phrase.pred_subject_in_clust = torch.ones_like(negative_subject_in_crowd)
    #     negative_to_pred_phrase.pred_object_in_clust = torch.ones_like(negative_object_in_crowd)
    #     negative_to_pred_phrase.confidence = torch.zeros_like(negative_to_pred_phrase.pred_subject_classes)
    #
    #     if self.print_size:
    #         print("positive to pred phrase", positive_to_pred_phrase_idx_gt_phrase_idx[0].shape[0],positive_to_pred_phrase_how_percent_subobj_gt_matched_topk.shape[0])
    #         print("negative to pred phrase", negative_to_pred_phrase_idx_gt_phrase_idx[0].shape[0],len(negative_to_pred_phrase))
    #         print("gt sub",gt_phrase.gt_subject_ids_list)
    #         print("pred sub",positive_to_pred_phrase.pred_subject_ids_list)
    #         print("gt obj",gt_phrase.gt_object_ids_list)
    #         print("pred obj",positive_to_pred_phrase.pred_object_ids_list)
    #         print("pred->gt rela",positive_to_pred_phrase_idx_gt_phrase_idx[1][positive_to_pred_phrase_select_by_mask_topk])
    #
    #     to_pred_phrase = Instances.cat([positive_to_pred_phrase, negative_to_pred_phrase])
    #     to_pred_phrase.pred_subject_crowd_ids = torch.arange(0, len(to_pred_phrase)).to(self.device).long()
    #     to_pred_phrase.pred_object_crowd_ids = torch.arange(len(to_pred_phrase), len(to_pred_phrase) * 2).to(self.device).long()
    #
    #     to_pred_crowd = Instances(to_pred_phrase.image_size)
    #     to_pred_crowd.pred_boxes=Boxes.cat([to_pred_phrase.pred_subject_boxes,to_pred_phrase.pred_object_boxes])
    #     to_pred_crowd.pred_instance_ids_list=torch.cat([to_pred_phrase.pred_subject_ids_list,to_pred_phrase.pred_object_ids_list])
    #     to_pred_crowd.pred_in_clust = torch.cat([to_pred_phrase.pred_subject_in_clust,to_pred_phrase.pred_object_in_clust])
    #     return to_pred_crowd, to_pred_phrase

    def generate_to_pred_phrases_from_to_pred_crowds(self,pred_instances,to_pred_crowds,gt_crowds,gt_phrases, training=True):
        to_pred_phrases=[]
        af_to_pred_crowds=[]
        for image_i in range(len(pred_instances)):
            pred_instance = pred_instances[image_i]
            to_pred_crowd = to_pred_crowds[image_i]
            if training:
                gt_crowd = gt_crowds[image_i]
                gt_phrase = gt_phrases[image_i]
            else:
                gt_crowd = None
                gt_phrase = None
            to_pred_crowd, to_pred_phrase = self.single_area_generate_to_pred_phrases_from_to_pred_crowds(pred_instance,to_pred_crowd,gt_crowd,gt_phrase,training)
            to_pred_phrases.append(to_pred_phrase)
            af_to_pred_crowds.append(to_pred_crowd)
        return af_to_pred_crowds, to_pred_phrases

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_seg_image(self, seg_image_tensor_list):
        images = [x/self.num_classes for x in seg_image_tensor_list]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

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

    def fullize_mask(self, instances):
        for instance_per_image in instances:
            height, width = instance_per_image.image_size
            fullize_mask(instance_per_image, height, width)

@META_ARCH_REGISTRY.register()
class GeneralizedRCNNWithPreCrowdProposalAllEncode(GeneralizedRCNN):

    def __init__(self, cfg):
        super(GeneralizedRCNNWithPreCrowdProposalAllEncode, self).__init__(cfg)
        self.crowd_proposal_generator = build_crowd_proposal_generator(cfg, self.backbone.output_shape(),type="crowd")
        self.to(self.device)

    def forward(self, batched_inputs, iteration, mode, training, print_size=False):  # batched_imputs['height'] and batched_imputs['width'] is real
        self.print_size = print_size
        losses = {}
        results=[]
        metrics={}

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if mode=="crowd":
            if training:
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                else:
                    gt_instances = None
                for image_i in range(len(gt_instances)):
                    gt_instances[image_i].pred_classes = gt_instances[image_i].gt_classes
                    gt_instances[image_i].pred_boxes = gt_instances[image_i].gt_boxes
                    gt_instances[image_i].pred_full_masks = gt_instances[image_i].gt_masks.tensor
                # if self.proposal_generator:
                #     proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                # else:
                #     print("need proposal generator")
                #     exit()
                # instance_detections, instance_local_features_roi, instance_local_features, instance_detector_loss = self.roi_heads(images, features, proposals, None,"instance",False)
                # self.fullize_mask(instance_detections)
                # if self.instance_encoder:
                #     instance_local_features = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
                # print("instance",len(instance_detections[0]),"gt instance",len(gt_instances[0]))

                if "crowds" in batched_inputs[0]:
                    gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
                else:
                    gt_crowds = None
                if self.crowd_proposal_generator:
                    crowd_proposals, crowd_proposal_losses, crowd_ctrness_pred, crowd_global_features = self.crowd_proposal_generator(images, features, gt_crowds, True)
                else:
                    assert "crowd_proposals" in batched_inputs[0]
                    crowd_proposals = [x["crowd_proposals"].to(self.device) for x in batched_inputs]
                    crowd_proposal_losses = {}
                losses.update(crowd_proposal_losses)
                # if crowd_proposals[0].has("proposal_boxes"):
                #     for image_i in range(len(crowd_proposals)):
                #         crowd_proposals[image_i].pred_boxes = crowd_proposals[image_i].proposal_boxes
                # pred_crowd_pred_instance_dict_list, max_len_list = self.generate_crowd_instance_dict(gt_instances, crowd_proposals)
                # to_pred_crowds = []
                # crowd_match_losses = []
                # for image_i in range(len(gt_instances)):
                #     to_pred_crowd, crowd_match_loss = self.single_area_generate_to_pred_crowds_for_to_pred_phrases(gt_instances[image_i], pred_crowd_pred_instance_dict_list[image_i], max_len_list[image_i], gt_crowds[image_i], training,compute_crowd_loss=True,pred_crowd=crowd_proposals[image_i])
                #     to_pred_crowds.append(to_pred_crowd)
                #     crowd_match_losses.append(crowd_match_loss)
                # losses['loss_crowd_match']=torch.mean(torch.stack(crowd_match_losses))

                # if gt_crowds:
                #     print("crowd proposal", len(crowd_proposals[0]), "to_pred_crowds", len(to_pred_crowds[0]),"gt crowds", len(gt_crowds[0]))
                # else:
                #     print("crowd proposal", len(crowd_proposals[0]), "to_pred_crowds", len(to_pred_crowds[0]))
            else:
                if self.proposal_generator:
                    proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                else:
                    print("need proposal generator")
                    exit()
                instance_detections, instance_local_features_roi, instance_local_features, instance_detector_loss = self.roi_heads(images, features, proposals, None,"instance",False)
                if self.instance_encoder:
                    instance_local_features = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
                print("instance", len(instance_detections[0]))

                if len(instance_detections[0])>0:
                    if self.crowd_proposal_generator:
                        crowd_proposals, crowd_proposal_losses, crowd_ctrness_pred, crowd_global_features = self.crowd_proposal_generator(images, features, None, training)
                    else:
                        assert "crowd_proposals" in batched_inputs[0]
                        crowd_proposals = [x["crowd_proposals"].to(self.device) for x in batched_inputs]
                        crowd_proposal_losses = {}
                    if crowd_proposals[0].has("proposal_boxes"):
                        for image_i in range(len(crowd_proposals)):
                            crowd_proposals[image_i].pred_boxes = crowd_proposals[image_i].proposal_boxes
                    crowd_proposals=self.filter_small_proposals(instance_detections,crowd_proposals)
                    print("crowd proposals", len(crowd_proposals[0]))
                    crowd_results = self._postprocess(crowd_proposals, batched_inputs, images.image_sizes,"crowd")
                    for i, r in enumerate(crowd_results):
                        if i >= len(results):
                            results.append({})
                        results[i].update(crowd_results[i])

        if mode=="relation":
            relation_results=None

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            if "crowds" in batched_inputs[0]:
                gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
            else:
                gt_crowds = None
            if "phrases" in batched_inputs[0]:
                gt_phrases = [x["phrases"].to(self.device) for x in batched_inputs]
            else:
                gt_phrases = None

            if self.use_gt_instance:
                for image_i in range(len(gt_instances)):
                    gt_instances[image_i].pred_classes = gt_instances[image_i].gt_classes
                    gt_instances[image_i].pred_boxes = gt_instances[image_i].gt_boxes
                    gt_instances[image_i].pred_full_masks = gt_instances[image_i].gt_masks.tensor
                instance_detections=gt_instances
                instance_local_features_roi, instance_local_features = self.roi_heads.generate_instance_box_features(features, instance_detections)
            else:
                if self.proposal_generator:
                    proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                else:
                    print("need proposal generator")
                    exit()
                instance_detections, instance_local_features_roi, instance_local_features, instance_detector_loss = self.roi_heads(images, features, proposals, None,"instance",False)
                self.fullize_mask(instance_detections)
                if training and (not self.use_gt_instance):
                    self.replace_gt_instance_with_pred_instances(instance_detections,gt_instances,gt_crowds,gt_phrases)

            if self.instance_encoder:
                instance_local_features_encode = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
            else:
                instance_local_features_encode=instance_local_features

            if self.print_size:
                if gt_instances:
                    print("instance",len(instance_detections[0]),"gt instance",len(gt_instances[0]))
                    print(instance_detections[0].pred_classes)
                else:
                    print("instance", len(instance_detections[0]))

            if len(instance_detections[0])>0:
                if self.crowd_proposal_generator:
                    crowd_proposals, crowd_proposal_losses, crowd_ctrness_pred, crowd_global_features = self.crowd_proposal_generator(images, features, None, training=False)
                else:
                    assert "crowd_proposals" in batched_inputs[0]
                    crowd_proposals = [x["crowd_proposals"].to(self.device) for x in batched_inputs]
                    crowd_proposal_losses = {}
                losses.update(crowd_proposal_losses)
                if crowd_proposals[0].has("proposal_boxes"):
                    for image_i in range(len(crowd_proposals)):
                        crowd_proposals[image_i].pred_boxes = crowd_proposals[image_i].proposal_boxes
                crowd_proposals = self.filter_small_proposals(instance_detections, crowd_proposals)

                crowd_proposals_all=Instances(instance_detections[0].image_size)
                crowd_proposals_all.pred_boxes=Boxes(torch.cat([crowd_proposals[0].pred_boxes.tensor,torch.Tensor([[0,0,instance_detections[0].image_size[1],instance_detections[0].image_size[0]]]).to(self.device)]))
                # crowd_proposals_all.pred_boxes = Boxes(torch.Tensor([[0, 0, instance_detections[0].image_size[1], instance_detections[0].image_size[0]]]).to(self.device))
                crowd_proposals[0]=crowd_proposals_all

                to_pred_crowds=[]
                for image_i in range(len(instance_detections)):
                    if False:
                        to_pred_crowd = self.single_area_generate_to_pred_crowds_for_to_pred_phrases(instance_detections[image_i], pred_crowd_pred_instance_dict_list[image_i],max_len_list[image_i], gt_crowds[image_i], training)
                    if training:
                        to_pred_crowd = self.gt_single_area_generate_to_pred_crowds_for_to_pred_phrases(instance_detections[image_i], crowd_proposals[image_i], gt_crowds[image_i],training)
                    else:
                        to_pred_crowd = self.gt_single_area_generate_to_pred_crowds_for_to_pred_phrases(instance_detections[image_i], crowd_proposals[image_i], None,training)

                    to_pred_crowds.append(to_pred_crowd)
                if self.print_size:
                    if gt_crowds:
                        print("crowd proposal", len(crowd_proposals[0]), "to_pred_crowds", len(to_pred_crowds[0]), "gt crowds", len(gt_crowds[0]))
                    else:
                        print("crowd proposal", len(crowd_proposals[0]),"to_pred_crowds", len(to_pred_crowds[0]))

                to_pred_crowds, to_pred_phrases = self.generate_to_pred_phrases_from_to_pred_crowds(instance_detections, to_pred_crowds, gt_crowds, gt_phrases,training)
                if self.print_size:
                    if gt_phrases:
                        print("to_pred_phrases", len(to_pred_phrases[0]),"gt phrases", len(gt_phrases[0]))
                    else:
                        print("to_pred_phrases", len(to_pred_phrases[0]))

                instance_detections[0].remove("pred_full_masks")

                if len(to_pred_phrases[0])>0:
                    to_pred_crowd_local_features_encode = None
                    to_pred_phrase_local_features_encode = None
                    if "crowd" in self.relation_head_list:
                        to_pred_crowd_local_features_roi, to_pred_crowd_local_features = self.roi_heads.generate_instance_box_features(
                            features, to_pred_crowds)
                        if self.instance_encoder:
                            to_pred_crowd_local_features_encode = [
                                self.instance_encoder(to_pred_crowd_local_feature) for to_pred_crowd_local_feature
                                in to_pred_crowd_local_features]
                        else:
                            to_pred_crowd_local_features_encode = to_pred_crowd_local_features
                    if "phrase" in self.relation_head_list:

                        to_pred_phrase_local_features_roi, to_pred_phrase_local_features = self.roi_heads.generate_instance_box_features(
                            features, to_pred_phrases)
                        if self.instance_encoder:
                            to_pred_phrase_local_features_encode = [
                                self.instance_encoder(to_pred_phrase_local_feature) for to_pred_phrase_local_feature
                                in to_pred_phrase_local_features]
                        else:
                            to_pred_phrase_local_features_encode = to_pred_phrase_local_features
                    relation_results, rela_losses_reduced, rela_metrics = self.relation_heads(instance_detections,
                                                                                              instance_local_features_encode,
                                                                                              to_pred_crowds,
                                                                                              to_pred_crowd_local_features_encode,
                                                                                              None,
                                                                                              to_pred_phrases,
                                                                                              to_pred_phrase_local_features_encode,
                                                                                              training=training,
                                                                                              iteration=iteration)

                    losses.update(rela_losses_reduced)
                else:
                    losses = {}
            else:
                losses={}

            if not training:
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes,"instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    if relation_results:
                        results[i]['phrases']=relation_results[i]
                    results[i].update(instance_results[i])

        return results, losses, metrics

    def filter_small_proposals(self, pred_instances, proposals):
        new_proposals=[]
        for image_i in range(len(proposals)):
            instance_boxes=pred_instances[image_i].pred_boxes.tensor
            proposal_boxes=proposals[image_i].pred_boxes.tensor
            instance_widths=(instance_boxes[:, 2] - instance_boxes[:, 0]).unsqueeze(0).repeat(proposal_boxes.shape[0],1)
            instance_heights=(instance_boxes[:, 3] - instance_boxes[:, 1]).unsqueeze(0).repeat(proposal_boxes.shape[0],1)
            proposal_width = (proposal_boxes[:, 2] - proposal_boxes[:, 0]).unsqueeze(1).repeat(1,instance_boxes.shape[0])
            proposal_height = (proposal_boxes[:, 3] - proposal_boxes[:, 1]).unsqueeze(1).repeat(1,instance_boxes.shape[0])
            proposal_instance_ios = compute_boxes_io(proposal_boxes, instance_boxes)
            proposal_select=torch.where(torch.sum((proposal_instance_ios > 0)*(proposal_width > instance_widths)*(proposal_height > instance_heights),dim=1)>0)
            new_proposals.append(proposals[image_i][proposal_select])
        return new_proposals

    def generate_crowd_instance_dict(self, pred_instances, pred_crowds):
        crowd_instance_dict_list=[]
        max_len_list=[]
        for image_i in range(len(pred_crowds)):
            pred_instance=pred_instances[image_i]
            pred_crowd=pred_crowds[image_i]

            crowd_instance_io = compute_boxes_io(pred_crowd.pred_boxes.tensor, pred_instance.pred_boxes.tensor)
            crowd_ids, instance_ids = torch.where(crowd_instance_io > 0.5)
            instance_classes = pred_instance.pred_classes[instance_ids]

            max_len = 0
            crowd_instance=defaultdict(list)
            for crowd_id, instance_id, instance_class in zip(crowd_ids, instance_ids, instance_classes):
                key = str(crowd_id.item()) + "_" + str(instance_class.item())
                crowd_instance[key].append(instance_id)
                max_len = max(max_len, len(crowd_instance[key]))

            crowd_instance_dict_list.append(crowd_instance)
            max_len_list.append(max_len)
        return crowd_instance_dict_list,max_len_list

    def gt_single_area_generate_to_pred_crowds_for_to_pred_phrases(self,pred_instance,pred_crowd,gt_crowd,training=True,compute_crowd_loss=False):
        # print(pred_instance.image_size)
        # print(pred_instance.pred_boxes)

        to_pred_crowd=Instances(pred_instance.image_size)
        pred_instance_boxes_pad0 = torch.cat([pred_instance.pred_boxes.tensor,torch.Tensor([[pred_instance.image_size[1],pred_instance.image_size[0],0,0]]).to(self.device)])

        class_set=torch.unique(pred_instance.pred_classes)

        to_pred_crowd_raw_crowd_ids=torch.Tensor(range(len(pred_crowd))).to(self.device).unsqueeze(1).repeat(1,class_set.shape[0]).flatten().long()
        to_pred_crowd_classes=class_set.unsqueeze(0).repeat(len(pred_crowd),1).flatten()

        to_pred_crowd_instance_ids_list=torch.Tensor(range(len(pred_instance))).to(self.device).unsqueeze(0).repeat(to_pred_crowd_raw_crowd_ids.shape[0],1).long()
        to_pred_crowd_instance_classes_list=pred_instance.pred_classes.unsqueeze(0).repeat(to_pred_crowd_raw_crowd_ids.shape[0],1)

        to_pred_crowd_boxes=pred_crowd.pred_boxes.tensor[to_pred_crowd_raw_crowd_ids]
        crowd_instance_ios = compute_boxes_io(to_pred_crowd_boxes, pred_instance.pred_boxes.tensor)
        crowd_class_instance_class_hit=to_pred_crowd_instance_classes_list == to_pred_crowd_classes.unsqueeze(1).repeat(1, len(pred_instance))
        matched_crowd_instance = crowd_instance_ios * (crowd_class_instance_class_hit)
        to_pred_crowd_instance_ids_list[matched_crowd_instance<self.positive_box_threshold]=-1
        matched_crowd_instance[matched_crowd_instance<self.positive_box_threshold] = 0
        to_pred_crowd_instance_ids_list, _ = torch.sort(to_pred_crowd_instance_ids_list, dim=1,descending=True)

        unique_to_pred_crowd_instance_ids_list = torch.unique(to_pred_crowd_instance_ids_list,dim=0)
        unique_to_pred_crowd_instance_ids_list = unique_to_pred_crowd_instance_ids_list[torch.sum(unique_to_pred_crowd_instance_ids_list>=0,dim=1)>0]
        to_pred_crowd_pred_classes = pred_instance.pred_classes[unique_to_pred_crowd_instance_ids_list[:, 0]]
        # print(unique_to_pred_crowd_instance_ids_list[0])
        # print(pred_instance_boxes_pad0[unique_to_pred_crowd_instance_ids_list][0])
        # print(union_box_matrix(pred_instance_boxes_pad0[unique_to_pred_crowd_instance_ids_list])[0])
        to_pred_crowd_pred_boxes = union_box_matrix(pred_instance_boxes_pad0[unique_to_pred_crowd_instance_ids_list])
        to_pred_crowd_pred_instance_lens = torch.sum(unique_to_pred_crowd_instance_ids_list >= 0, dim=1)

        max_len=torch.max(to_pred_crowd_pred_instance_lens)
        unique_to_pred_crowd_instance_ids_list=unique_to_pred_crowd_instance_ids_list[:,:max_len]
        to_pred_crowd.pred_instance_ids_list = unique_to_pred_crowd_instance_ids_list.long()
        to_pred_crowd.pred_classes = to_pred_crowd_pred_classes.long()
        to_pred_crowd.pred_instance_lens = to_pred_crowd_pred_instance_lens.long()
        to_pred_crowd.pred_boxes = Boxes(to_pred_crowd_pred_boxes)
        to_pred_crowd.pred_in_clust = torch.ones_like(unique_to_pred_crowd_instance_ids_list)

        if not training:
            _,to_pred_crowd_select=torch.topk(to_pred_crowd_pred_instance_lens,min(min(int(2000/len(pred_instance)),100),to_pred_crowd_pred_instance_lens.shape[0]))
            to_pred_crowd=to_pred_crowd[to_pred_crowd_select]
        # if training:
        #     to_pred_crowd_gt_crowd_class_hit=to_pred_crowd.pred_classes.unsqueeze(1).repeat(1,len(gt_crowd))==gt_crowd.gt_classes.unsqueeze(0).repeat(to_pred_crowd.pred_classes.shape[0],1) # crowd,gt_crowd
        #     pred_ins_ids=to_pred_crowd.pred_instance_ids_list.unsqueeze(1).repeat(1,len(gt_crowd),1).unsqueeze(3).repeat(1,1,1,gt_crowd.gt_instance_ids_list.shape[1]) # crowd,gt_crowd,pred_instance,gt_instance
        #     pred_ins_ids[pred_ins_ids==-1]=-2
        #     gt_ins_ids = gt_crowd.gt_instance_ids_list.unsqueeze(1).repeat(1, len(pred_instance), 1).unsqueeze(0).repeat(to_pred_crowd.pred_instance_ids_list.shape[0],1,1,1) # crowd,gt_crowd,instance,gt_instance
        #     to_pred_crowd_gt_mask_ios_lists_pred_instance_mask = (torch.sum(pred_ins_ids == gt_ins_ids, dim=3) > 0).float() # crowd,gt_crowd,instance
        #     to_pred_crowd_gt_mask_ios_lists_pred_instance_mask=to_pred_crowd_gt_mask_ios_lists_pred_instance_mask * to_pred_crowd_gt_crowd_class_hit.unsqueeze(2)
        #     to_pred_crowd_pred_mask_ios_list_gt_mask=(torch.sum(to_pred_crowd_gt_mask_ios_lists_pred_instance_mask,dim=2).float()/gt_crowd.gt_instance_lens.unsqueeze(0).repeat(to_pred_crowd_gt_mask_ios_lists_pred_instance_mask.shape[0],1)) * to_pred_crowd_gt_crowd_class_hit
        #     to_pred_crowd.pred_mask_ios_list_gt_mask=to_pred_crowd_pred_mask_ios_list_gt_mask
        #     to_pred_crowd.gt_mask_ios_lists_pred_instance_mask=to_pred_crowd_gt_mask_ios_lists_pred_instance_mask
        return to_pred_crowd

@META_ARCH_REGISTRY.register()
class GeneralizedRCNNWithPrePhraseProposalAllEncode(GeneralizedRCNN):

    def __init__(self, cfg):
        super(GeneralizedRCNNWithPrePhraseProposalAllEncode, self).__init__(cfg)
        self.phrase_proposal_generator = build_phrase_proposal_generator(cfg, self.backbone.output_shape(),type="phrase")
        self.to(self.device)

    def forward(self, batched_inputs, iteration, mode, training, print_size=False):  # batched_imputs['height'] and batched_imputs['width'] is real
        self.print_size = print_size
        losses = {}
        results=[]
        metrics={}

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if mode=="phrase":
            if training:
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                else:
                    gt_instances = None
                for image_i in range(len(gt_instances)):
                    gt_instances[image_i].pred_classes = gt_instances[image_i].gt_classes
                    gt_instances[image_i].pred_boxes = gt_instances[image_i].gt_boxes
                    gt_instances[image_i].pred_full_masks = gt_instances[image_i].gt_masks.tensor
                # if self.proposal_generator:
                #     proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                # else:
                #     print("need proposal generator")
                #     exit()
                # instance_detections, instance_local_features_roi, instance_local_features, instance_detector_loss = self.roi_heads(images, features, proposals, None,"instance",False)
                # if self.instance_encoder:
                #     instance_local_features=self.instance_encoder(instance_local_features)
                # self.fullize_mask(instance_detections)
                # print("instance",len(instance_detections[0]),"gt instance",len(gt_instances[0]))
                if "crowds" in batched_inputs[0]:
                    gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
                else:
                    gt_crowds = None
                if "phrases" in batched_inputs[0]:
                    gt_phrases = [x["phrases"].to(self.device) for x in batched_inputs]
                else:
                    gt_phrases = None
                if self.phrase_proposal_generator:
                    phrase_proposals, phrase_proposal_losses, phrase_ctrness_pred, phrase_global_features = self.phrase_proposal_generator(images, features, gt_phrases, True)
                else:
                    assert "phrase_proposals" in batched_inputs[0]
                    phrase_proposals = [x["phrase_proposals"].to(self.device) for x in batched_inputs]
                    phrase_proposal_losses = {}
                losses.update(phrase_proposal_losses)
                # if phrase_proposals[0].has("proposal_boxes"):
                #     for image_i in range(len(phrase_proposals)):
                #         phrase_proposals[image_i].pred_boxes = phrase_proposals[image_i].proposal_boxes
                # pred_phrase_pred_instance_dict_list, max_len_list = self.generate_phrase_instance_dict(gt_instances,phrase_proposals)
                # to_pred_crowds = []
                # to_pred_phrases = []
                # crowd_match_losses = []
                # for image_i in range(len(gt_instances)):
                #     image_to_pred_crowds = []
                #     image_to_pred_phrases = []
                #     image_to_pred_crowd_num = 0
                #     pred_phrase_pred_instance_dict = pred_phrase_pred_instance_dict_list[image_i]
                #     for phrase_id in pred_phrase_pred_instance_dict:
                #         phrase_to_pred_crowd, crowd_match_loss = self.single_area_generate_to_pred_crowds_for_to_pred_phrases(gt_instances[image_i], pred_phrase_pred_instance_dict[phrase_id], max_len_list[image_i],gt_crowds[image_i], training,compute_crowd_loss=True,pred_crowd=phrase_proposals[image_i])
                #         crowd_match_losses.append(crowd_match_loss)
                #         phrase_to_pred_phrase = self.single_area_generate_to_pred_phrases_from_to_pred_crowds(gt_instances[image_i], phrase_to_pred_crowd, gt_crowds[image_i], gt_phrases[image_i])
                #         phrase_to_pred_phrase.pred_subject_crowd_ids = phrase_to_pred_phrase.pred_subject_crowd_ids + image_to_pred_crowd_num
                #         phrase_to_pred_phrase.pred_object_crowd_ids = phrase_to_pred_phrase.pred_object_crowd_ids + image_to_pred_crowd_num
                #         image_to_pred_crowds.append(phrase_to_pred_crowd)
                #         image_to_pred_phrases.append(phrase_to_pred_phrase)
                #         image_to_pred_crowd_num += len(phrase_to_pred_crowd)
                #     to_pred_crowds.append(Instances.cat(image_to_pred_crowds))
                #     to_pred_phrases.append(Instances.cat(image_to_pred_phrases))
                # losses['loss_crowd_match'] = torch.mean(torch.stack(crowd_match_losses))
                # if gt_crowds:
                #     print("to_pred_crowds", len(to_pred_crowds[0]), "gt crowd", len(gt_crowds[0]))
                # else:
                #     print("to_pred_crowds", len(to_pred_crowds[0]))
                # print("to_pred_phrases", len(to_pred_phrases[0]))

            else:
                if self.proposal_generator:
                    proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                else:
                    print("need proposal generator")
                    exit()
                instance_detections, instance_local_features_roi, instance_local_features, instance_detector_loss = self.roi_heads(images, features, proposals, None,"instance",False)
                if self.instance_encoder:
                    instance_local_features = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
                print("instance", len(instance_detections[0]))

                if len(instance_detections[0])>0:
                    if self.phrase_proposal_generator:
                        phrase_proposals, phrase_proposal_losses, phrase_ctrness_pred, phrase_global_features = self.phrase_proposal_generator(images, features, None, training)
                    else:
                        assert "phrase_proposals" in batched_inputs[0]
                        phrase_proposals = [x["phrase_proposals"].to(self.device) for x in batched_inputs]
                        phrase_proposal_losses = {}
                    if phrase_proposals[0].has("proposal_boxes"):
                        for image_i in range(len(phrase_proposals)):
                            phrase_proposals[image_i].pred_boxes = phrase_proposals[image_i].proposal_boxes
                    phrase_proposals=self.filter_small_proposals(instance_detections,phrase_proposals)
                    print("phrase proposals", len(phrase_proposals[0]))
                    phrase_results = self._postprocess(phrase_proposals, batched_inputs, images.image_sizes,"phrase")
                    for i, r in enumerate(phrase_results):
                        if i >= len(results):
                            results.append({})
                        results[i].update(phrase_results[i])

        if mode=="relation":
            relation_results=None

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            if "crowds" in batched_inputs[0]:
                gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
            else:
                gt_crowds = None
            if "phrases" in batched_inputs[0]:
                gt_phrases = [x["phrases"].to(self.device) for x in batched_inputs]
            else:
                gt_phrases = None
            if self.use_gt_instance:
                for image_i in range(len(gt_instances)):
                    gt_instances[image_i].pred_classes = gt_instances[image_i].gt_classes
                    gt_instances[image_i].pred_boxes = gt_instances[image_i].gt_boxes
                    gt_instances[image_i].pred_full_masks = gt_instances[image_i].gt_masks.tensor
                instance_detections=gt_instances
                instance_local_features_roi, instance_local_features = self.roi_heads.generate_instance_box_features(features, instance_detections)
            else:
                if self.proposal_generator:
                    instance_proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                else:
                    print("need proposal generator")
                    exit()
                instance_detections, instance_local_features_roi, instance_local_features, instance_detector_loss = self.roi_heads(images, features, instance_proposals, None,"instance",False)
                self.fullize_mask(instance_detections)
                if training and not (self.use_gt_instance):
                    self.replace_gt_instance_with_pred_instances(instance_detections,gt_instances,gt_crowds,gt_phrases)
            if self.instance_encoder:
                instance_local_features_encode = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
            else:
                instance_local_features_encode=instance_local_features
            if self.print_size:
                if gt_instances:
                    print("instance",len(instance_detections[0]),"gt instance",len(gt_instances[0]))
                    # print(instance_detections[0].pred_classes)
                else:
                    print("instance", len(instance_detections[0]))

            if len(instance_detections[0])>0:
                if self.phrase_proposal_generator:
                    phrase_proposals, phrase_proposal_losses, phrase_ctrness_pred, phrase_global_features = self.phrase_proposal_generator(images, features, None, training=False)
                else:
                    assert "phrase_proposals" in batched_inputs[0]
                    phrase_proposals = [x["phrase_proposals"].to(self.device) for x in batched_inputs]
                    phrase_proposal_losses = {}
                losses.update(phrase_proposal_losses)
                if phrase_proposals[0].has("proposal_boxes"):
                    for image_i in range(len(phrase_proposals)):
                        phrase_proposals[image_i].pred_boxes = phrase_proposals[image_i].proposal_boxes
                phrase_proposals = self.filter_small_proposals(instance_detections, phrase_proposals)
                phrase_proposals_all=Instances(instance_detections[0].image_size)
                phrase_proposals_all.pred_boxes=Boxes(torch.cat([phrase_proposals[0].pred_boxes.tensor,torch.Tensor([[0,0,instance_detections[0].image_size[1],instance_detections[0].image_size[0]]]).to(self.device)]))
                # phrase_proposals_all.pred_boxes=Boxes(torch.Tensor([[0,0,instance_detections[0].image_size[1],instance_detections[0].image_size[0]]]).to(self.device))
                phrase_proposals[0]=phrase_proposals_all
                if self.print_size:
                    if gt_phrases:
                        print("phrase proposal", len(phrase_proposals[0]),"gt phrase",len(gt_phrases[0]))
                    else:
                        print("phrase proposal", len(phrase_proposals[0]))

                if False:
                    pred_phrase_pred_instance_dict_list, max_len_list = self.generate_phrase_instance_dict(instance_detections, phrase_proposals)
                    to_pred_crowds=[]
                    to_pred_phrases=[]
                    for image_i in range(len(gt_instances)):
                        image_to_pred_crowds = []
                        image_to_pred_phrases = []
                        image_to_pred_crowd_num=0
                        pred_phrase_pred_instance_dict = pred_phrase_pred_instance_dict_list[image_i]
                        for phrase_id in pred_phrase_pred_instance_dict:
                            phrase_to_pred_crowd = self.single_area_generate_to_pred_crowds_for_to_pred_phrases(instance_detections[image_i],pred_phrase_pred_instance_dict[phrase_id],max_len_list[image_i],gt_crowds[image_i],training)
                            phrase_to_pred_phrase = self.single_area_generate_to_pred_phrases_from_to_pred_crowds(instance_detections[image_i],phrase_to_pred_crowd,gt_crowds[image_i],gt_phrases[image_i],training)
                            phrase_to_pred_phrase.pred_subject_crowd_ids=phrase_to_pred_phrase.pred_subject_crowd_ids+image_to_pred_crowd_num
                            phrase_to_pred_phrase.pred_object_crowd_ids = phrase_to_pred_phrase.pred_object_crowd_ids + image_to_pred_crowd_num
                            image_to_pred_crowds.append(phrase_to_pred_crowd)
                            image_to_pred_phrases.append(phrase_to_pred_phrase)
                            image_to_pred_crowd_num+=len(phrase_to_pred_crowd)
                    to_pred_crowds.append(Instances.cat(image_to_pred_crowds))
                    image_to_pred_phrase=Instances.cat(image_to_pred_phrases)
                    _,select_topk=torch.topk(image_to_pred_phrase.to_pred_phrase_subobj_crowd_pred_mask_ios_gt_mask_topk,min(1000,len(image_to_pred_phrase)))
                    to_pred_phrases.append(image_to_pred_phrase[select_topk])
                else:
                    to_pred_crowds=[]
                    to_pred_phrases=[]
                    for image_i in range(len(instance_detections)):
                        if training:
                            to_pred_crowd, to_pred_phrase=self.gt_single_area_generate_to_pred_phrases(instance_detections[image_i],phrase_proposals[image_i],gt_phrases[image_i],training)
                        else:
                            to_pred_crowd, to_pred_phrase = self.gt_single_area_generate_to_pred_phrases(instance_detections[image_i],phrase_proposals[image_i],None, training)
                        to_pred_crowds.append(to_pred_crowd)
                        to_pred_phrases.append(to_pred_phrase)

                if self.print_size:
                    if len(to_pred_crowds)>0:
                        if gt_crowds:
                            print("to_pred_crowds",len(to_pred_crowds[0]),"gt crowd",len(gt_crowds[0]))
                        else:
                            print("to_pred_crowds", len(to_pred_crowds[0]))
                    if len(to_pred_phrases)>0:
                        if gt_phrases:
                            print("to_pred_phrases",len(to_pred_phrases[0]),"gt phrase",len(gt_phrases[0]))
                        else:
                            print("to_pred_phrases", len(to_pred_phrases[0]))

                instance_detections[0].remove("pred_full_masks")

                if len(to_pred_phrases[0])>0:
                    to_pred_crowd_local_features_encode = None
                    to_pred_phrase_local_features_encode = None
                    if "crowd" in self.relation_head_list:
                        to_pred_crowd_local_features_roi, to_pred_crowd_local_features = self.roi_heads.generate_instance_box_features(
                            features, to_pred_crowds)
                        if self.instance_encoder:
                            to_pred_crowd_local_features_encode = [
                                self.instance_encoder(to_pred_crowd_local_feature) for to_pred_crowd_local_feature
                                in to_pred_crowd_local_features]
                        else:
                            to_pred_crowd_local_features_encode = to_pred_crowd_local_features
                    if "phrase" in self.relation_head_list:

                        to_pred_phrase_local_features_roi, to_pred_phrase_local_features = self.roi_heads.generate_instance_box_features(
                            features, to_pred_phrases)
                        if self.instance_encoder:
                            to_pred_phrase_local_features_encode = [
                                self.instance_encoder(to_pred_phrase_local_feature) for to_pred_phrase_local_feature
                                in to_pred_phrase_local_features]
                        else:
                            to_pred_phrase_local_features_encode = to_pred_phrase_local_features
                    relation_results, rela_losses_reduced, rela_metrics = self.relation_heads(instance_detections,
                                                                                              instance_local_features_encode,
                                                                                              to_pred_crowds,
                                                                                              to_pred_crowd_local_features_encode,
                                                                                              None,
                                                                                              to_pred_phrases,
                                                                                              to_pred_phrase_local_features_encode,
                                                                                              training=training,
                                                                                              iteration=iteration)
                    losses.update(rela_losses_reduced)
                else:
                    losses = {}
            else:
                losses={}
            if not training:
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes,"instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    if relation_results:
                        results[i]['phrases']=relation_results[i]
                    results[i].update(instance_results[i])

        return results, losses, metrics

    def filter_small_proposals(self, pred_instances, proposals):
        new_proposals=[]
        for image_i in range(len(proposals)):
            instance_boxes=pred_instances[image_i].pred_boxes.tensor
            proposal_boxes=proposals[image_i].pred_boxes.tensor
            instance_widths=(instance_boxes[:, 2] - instance_boxes[:, 0]).unsqueeze(0).repeat(proposal_boxes.shape[0],1)
            instance_heights=(instance_boxes[:, 3] - instance_boxes[:, 1]).unsqueeze(0).repeat(proposal_boxes.shape[0],1)
            proposal_width = (proposal_boxes[:, 2] - proposal_boxes[:, 0]).unsqueeze(1).repeat(1,instance_boxes.shape[0])
            proposal_height = (proposal_boxes[:, 3] - proposal_boxes[:, 1]).unsqueeze(1).repeat(1,instance_boxes.shape[0])
            proposal_instance_ios = compute_boxes_io(proposal_boxes, instance_boxes)
            proposal_select=torch.where(torch.sum((proposal_instance_ios > 0)*(proposal_width > instance_widths)*(proposal_height > instance_heights),dim=1)>0)
            new_proposals.append(proposals[image_i][proposal_select])
        return new_proposals

    def generate_phrase_instance_dict(self, pred_instances, pred_phrases):
        phrase_instance_dict_list=[]
        max_len_list=[]
        for image_i in range(len(pred_phrases)):
            pred_instance=pred_instances[image_i]
            pred_phrase=pred_phrases[image_i]

            phrase_instance_io = compute_boxes_io(pred_phrase.pred_boxes.tensor, pred_instance.pred_boxes.tensor)
            phrase_ids, instance_ids = torch.where(phrase_instance_io > 0.5)
            instance_classes = pred_instance.pred_classes[instance_ids]

            phrase_instance = defaultdict(dict)
            max_len = 0
            for phrase_id, instance_id, instance_class in zip(phrase_ids, instance_ids, instance_classes):
                phrase_id_item = str(phrase_id.item())
                instance_class_item = str(instance_class.item())
                inner_key = phrase_id_item+"_"+instance_class_item
                if inner_key not in phrase_instance[phrase_id_item]:
                    phrase_instance[phrase_id_item][inner_key] = []
                phrase_instance[phrase_id_item][inner_key].append(instance_id)
                max_len = max(max_len, len(phrase_instance[phrase_id_item][inner_key]))

            phrase_instance_dict_list.append(phrase_instance)
            max_len_list.append(max_len)
        return phrase_instance_dict_list,max_len_list

    def gt_single_area_generate_to_pred_phrases(self,pred_instance,pred_phrase,gt_phrase, training=True):
        # print(pred_instance.image_size)
        # print(pred_instance.pred_boxes)
        class_set = torch.unique(pred_instance.pred_classes)
        # print(class_set)
        subject_class_set = class_set.unsqueeze(1).repeat(1,class_set.shape[0]).flatten()
        object_class_set = class_set.unsqueeze(0).repeat(class_set.shape[0],1).flatten()
        pred_instance_boxes_pad0 = torch.cat([pred_instance.pred_boxes.tensor, torch.Tensor([[pred_instance.image_size[1],pred_instance.image_size[0],0,0]]).to(self.device)])

        to_pred_crowd = Instances(pred_instance.image_size)
        to_pred_phrase = Instances(pred_instance.image_size)

        to_pred_phrase_raw_phrase_ids = torch.Tensor(range(len(pred_phrase))).to(self.device).unsqueeze(1).repeat(1, class_set.shape[0]*class_set.shape[0]).flatten().long()
        to_pred_phrase_subject_classes = subject_class_set.unsqueeze(0).repeat(len(pred_phrase), 1).flatten()
        to_pred_phrase_object_classes = object_class_set.unsqueeze(0).repeat(len(pred_phrase), 1).flatten()
        to_pred_phrase_subject_ids_list = torch.Tensor(range(len(pred_instance))).to(self.device).unsqueeze(0).repeat(to_pred_phrase_raw_phrase_ids.shape[0], 1).long()
        to_pred_phrase_object_ids_list = torch.Tensor(range(len(pred_instance))).to(self.device).unsqueeze(0).repeat(to_pred_phrase_raw_phrase_ids.shape[0], 1).long()

        to_pred_phrase_instance_classes_list = pred_instance.pred_classes.unsqueeze(0).repeat(to_pred_phrase_raw_phrase_ids.shape[0], 1)
        to_pred_phrase_boxes = pred_phrase.pred_boxes.tensor[to_pred_phrase_raw_phrase_ids]

        phrase_instance_ios = compute_boxes_io(to_pred_phrase_boxes, pred_instance.pred_boxes.tensor)
        matched_phrase_subject_instance = phrase_instance_ios * (to_pred_phrase_instance_classes_list == to_pred_phrase_subject_classes.unsqueeze(1).repeat(1, len(pred_instance)))
        matched_phrase_object_instance = phrase_instance_ios * (to_pred_phrase_instance_classes_list == to_pred_phrase_object_classes.unsqueeze(1).repeat(1, len(pred_instance)))

        to_pred_phrase_subject_ids_list[matched_phrase_subject_instance < self.positive_box_threshold] = -1
        to_pred_phrase_object_ids_list[matched_phrase_object_instance < self.positive_box_threshold] = -1
        matched_phrase_subject_instance[matched_phrase_subject_instance < self.positive_box_threshold] = 0
        matched_phrase_object_instance[matched_phrase_object_instance < self.positive_box_threshold] = 0

        sub_len = torch.sum(to_pred_phrase_subject_ids_list >= 0,dim=1)
        obj_len = torch.sum(to_pred_phrase_object_ids_list >= 0, dim=1)
        sub_len[sub_len == 0] = 1
        obj_len[obj_len == 0] = 1
        matched_phrase_subject_instance_mean=torch.sum(matched_phrase_subject_instance,dim=1)/sub_len
        matched_phrase_object_instance_mean=torch.sum(matched_phrase_object_instance,dim=1)/obj_len
        to_pred_phrase_select = torch.where(matched_phrase_subject_instance_mean*matched_phrase_object_instance_mean > 0)[0]

        if to_pred_phrase_select.shape[0]>0:
            to_pred_phrase_subject_ids_list = torch.sort(to_pred_phrase_subject_ids_list[to_pred_phrase_select, :],dim=1,descending=True)[0].long()
            to_pred_phrase_object_ids_list = torch.sort(to_pred_phrase_object_ids_list[to_pred_phrase_select, :],dim=1,descending=True)[0].long()

            to_pred_phrase_subobj_ids_list = torch.cat([to_pred_phrase_subject_ids_list.unsqueeze(1),to_pred_phrase_object_ids_list.unsqueeze(1)],dim=1)
            unique_to_pred_phrase_subobj_ids_list = torch.unique(to_pred_phrase_subobj_ids_list,dim=0)

            unique_to_pred_phrase_subject_ids_list = unique_to_pred_phrase_subobj_ids_list[:,0,:]
            unique_to_pred_phrase_object_ids_list = unique_to_pred_phrase_subobj_ids_list[:, 1, :]

            to_pred_phrase.pred_subject_ids_list = unique_to_pred_phrase_subject_ids_list
            to_pred_phrase.pred_object_ids_list = unique_to_pred_phrase_object_ids_list

            to_pred_phrase.pred_subject_classes = pred_instance.pred_classes[unique_to_pred_phrase_subject_ids_list[:,0]].long()
            to_pred_phrase.pred_object_classes = pred_instance.pred_classes[unique_to_pred_phrase_object_ids_list[:,0]].long()

            to_pred_phrase.pred_subject_lens = torch.sum(unique_to_pred_phrase_subject_ids_list>=0,dim=1).long()
            to_pred_phrase.pred_object_lens = torch.sum(unique_to_pred_phrase_object_ids_list>=0,dim=1).long()
            # print(unique_to_pred_phrase_subject_ids_list[0])
            # print(pred_instance_boxes_pad0[unique_to_pred_phrase_subject_ids_list][0])
            # print(union_box_matrix(pred_instance_boxes_pad0[unique_to_pred_phrase_subject_ids_list])[0])
            to_pred_phrase.pred_subject_boxes = Boxes(union_box_matrix(pred_instance_boxes_pad0[unique_to_pred_phrase_subject_ids_list]))
            to_pred_phrase.pred_object_boxes = Boxes(union_box_matrix(pred_instance_boxes_pad0[unique_to_pred_phrase_object_ids_list]))
            to_pred_phrase.pred_boxes = Boxes(union_boxes(to_pred_phrase.pred_subject_boxes.tensor,to_pred_phrase.pred_object_boxes.tensor))
            to_pred_phrase.pred_subject_crowd_ids = torch.arange(0,len(to_pred_phrase)).to(self.device).long()
            to_pred_phrase.pred_object_crowd_ids = torch.arange(len(to_pred_phrase),len(to_pred_phrase)*2).to(self.device).long()
            to_pred_phrase.pred_subject_in_clust = torch.ones_like(to_pred_phrase.pred_subject_ids_list)
            to_pred_phrase.pred_object_in_clust = torch.ones_like(to_pred_phrase.pred_subject_ids_list)

            to_pred_crowd.pred_instance_ids_list = torch.cat([to_pred_phrase.pred_subject_ids_list,to_pred_phrase.pred_object_ids_list]).long()
            to_pred_crowd.pred_instance_lens = torch.cat([to_pred_phrase.pred_subject_lens,to_pred_phrase.pred_object_lens]).long()
            to_pred_crowd.pred_classes = torch.cat([to_pred_phrase.pred_subject_classes,to_pred_phrase.pred_object_classes]).long()
            to_pred_crowd.pred_boxes = Boxes.cat([to_pred_phrase.pred_subject_boxes,to_pred_phrase.pred_object_boxes])
            to_pred_crowd.pred_in_clust = torch.ones_like(to_pred_crowd.pred_instance_ids_list)
            # print(to_pred_phrase.pred_subject_ids_list)
            # print(to_pred_phrase.pred_object_ids_list)
            # print(to_pred_phrase.pred_subject_classes)
            # print(to_pred_phrase.pred_object_classes)

            if training:
                to_pred_phrase = self.single_area_choose_phrase_for_training(pred_instance,to_pred_phrase,gt_phrase)
            # if training:
            #     to_pred_phrase_gt_phrase_subject_class_hit = to_pred_phrase.pred_subject_classes.unsqueeze(1).repeat(1, len(gt_phrase)) == gt_phrase.gt_subject_classes.unsqueeze(0).repeat(to_pred_phrase.pred_subject_classes.shape[0],1)
            #     to_pred_phrase_gt_phrase_object_class_hit = to_pred_phrase.pred_object_classes.unsqueeze(1).repeat(1, len(gt_phrase)) == gt_phrase.gt_object_classes.unsqueeze(0).repeat(to_pred_phrase.pred_object_classes.shape[0],1)
            #     pred_subject_ids = to_pred_phrase.pred_subject_ids_list.unsqueeze(1).repeat(1, len(gt_phrase), 1).unsqueeze(3).repeat(1, 1, 1, gt_phrase.gt_subject_ids_list.shape[1])  # phrase,gt_phrase,pred_instance,gt_instance
            #     pred_object_ids = to_pred_phrase.pred_object_ids_list.unsqueeze(1).repeat(1, len(gt_phrase), 1).unsqueeze(3).repeat(1, 1, 1, gt_phrase.gt_object_ids_list.shape[1])  # phrase,gt_phrase,pred_instance,gt_instance
            #     pred_subject_ids[pred_subject_ids == -1] = -2
            #     pred_object_ids[pred_object_ids == -1] = -2
            #     gt_subject_ids = gt_phrase.gt_subject_ids_list.unsqueeze(1).repeat(1, len(pred_instance), 1).unsqueeze(0).repeat(to_pred_phrase.pred_subject_classes.shape[0], 1, 1, 1)  # phrase,gt_phrase,instance,gt_instance
            #     gt_object_ids = gt_phrase.gt_object_ids_list.unsqueeze(1).repeat(1, len(pred_instance), 1).unsqueeze(0).repeat(to_pred_phrase.pred_object_classes.shape[0], 1, 1, 1)  # phrase,gt_phrase,instance,gt_instance
            #     to_pred_phrase_gt_mask_ios_lists_pred_subject_mask = (torch.sum(pred_subject_ids == gt_subject_ids, dim=3) > 0).float()  # phrase,gt_phrase,instance
            #     to_pred_phrase_gt_mask_ios_lists_pred_object_mask = (torch.sum(pred_object_ids == gt_object_ids, dim=3) > 0).float()  # phrase,gt_phrase,instance
            #     # print("gt-instance ios mask")
            #     # print(to_pred_phrase_gt_mask_ios_lists_pred_subject_mask)
            #     # print(to_pred_phrase_gt_mask_ios_lists_pred_object_mask)
            #     # print("class hit")
            #     # print(to_pred_phrase_gt_phrase_subject_class_hit)
            #     # print(to_pred_phrase_gt_phrase_object_class_hit)
            #     to_pred_phrase_gt_mask_ios_lists_pred_subject_mask = to_pred_phrase_gt_mask_ios_lists_pred_subject_mask * to_pred_phrase_gt_phrase_subject_class_hit.unsqueeze(2)
            #     to_pred_phrase_gt_mask_ios_lists_pred_object_mask = to_pred_phrase_gt_mask_ios_lists_pred_object_mask * to_pred_phrase_gt_phrase_object_class_hit.unsqueeze(2)
            #     # print("gt-instance ios mask with class hit")
            #     # print(to_pred_phrase_gt_mask_ios_lists_pred_subject_mask)
            #     # print(to_pred_phrase_gt_mask_ios_lists_pred_object_mask)
            #     # print("gt_phrase len")
            #     # print(gt_phrase.gt_subject_lens.unsqueeze(0).repeat(to_pred_phrase_gt_mask_ios_lists_pred_subject_mask.shape[0],1))
            #     # print(gt_phrase.gt_subject_lens.unsqueeze(0).repeat(to_pred_phrase_gt_mask_ios_lists_pred_subject_mask.shape[0], 1))
            #     to_pred_phrase_pred_subject_mask_ios_list_gt_mask = (torch.sum(to_pred_phrase_gt_mask_ios_lists_pred_subject_mask,dim=2).float() / gt_phrase.gt_subject_lens.unsqueeze(0).repeat(to_pred_phrase_gt_mask_ios_lists_pred_subject_mask.shape[0],1)) * to_pred_phrase_gt_phrase_subject_class_hit
            #     to_pred_phrase_pred_object_mask_ios_list_gt_mask = (torch.sum(to_pred_phrase_gt_mask_ios_lists_pred_object_mask,dim=2).float() / gt_phrase.gt_object_lens.unsqueeze(0).repeat(to_pred_phrase_gt_mask_ios_lists_pred_object_mask.shape[0],1)) * to_pred_phrase_gt_phrase_object_class_hit
            #     # print("crowd ios ratio")
            #     # print(to_pred_phrase_pred_subject_mask_ios_list_gt_mask)
            #     # print(to_pred_phrase_pred_object_mask_ios_list_gt_mask)
            #     positive_to_pred_phrase_idx_gt_phrase_idx = torch.where(((to_pred_phrase_pred_subject_mask_ios_list_gt_mask >= self.positive_crowd_threshold)*(to_pred_phrase_pred_object_mask_ios_list_gt_mask >= self.positive_crowd_threshold))>0) #positive: phrase,gt_phrase
            #     negative_to_pred_phrase_idx_gt_phrase_idx = torch.where(((to_pred_phrase_pred_subject_mask_ios_list_gt_mask < self.positive_crowd_threshold) + (to_pred_phrase_pred_object_mask_ios_list_gt_mask < self.positive_crowd_threshold)) > 0)  # negative: phrase,gt_phrase
            #     # print("positive",positive_to_pred_phrase_idx_gt_phrase_idx)
            #     # print("negative",negative_to_pred_phrase_idx_gt_phrase_idx)
            #     positive_to_pred_phrase_pred_subject_mask_ios_list_gt_mask = to_pred_phrase_pred_subject_mask_ios_list_gt_mask[positive_to_pred_phrase_idx_gt_phrase_idx]
            #     positive_to_pred_phrase_pred_object_mask_ios_list_gt_mask = to_pred_phrase_pred_object_mask_ios_list_gt_mask[positive_to_pred_phrase_idx_gt_phrase_idx]
            #     negative_to_pred_phrase_pred_subject_mask_ios_list_gt_mask = to_pred_phrase_pred_subject_mask_ios_list_gt_mask[negative_to_pred_phrase_idx_gt_phrase_idx]
            #     negative_to_pred_phrase_pred_object_mask_ios_list_gt_mask = to_pred_phrase_pred_object_mask_ios_list_gt_mask[negative_to_pred_phrase_idx_gt_phrase_idx]
            #     # print("positive")
            #     # print(positive_to_pred_phrase_pred_subject_mask_ios_list_gt_mask)
            #     # print(positive_to_pred_phrase_pred_object_mask_ios_list_gt_mask)
            #     # print("negative")
            #     # print(negative_to_pred_phrase_pred_subject_mask_ios_list_gt_mask)
            #     # print(negative_to_pred_phrase_pred_object_mask_ios_list_gt_mask)
            #     # print(gt_phrase.gt_subject_ids_list)
            #     # print(gt_phrase.gt_object_ids_list)
            #     # print(gt_phrase.gt_subject_classes)
            #     # print(gt_phrase.gt_object_classes)
            #     # print()
            #     # print(to_pred_phrase[positive_to_pred_phrase_idx_gt_phrase_idx[0]].pred_subject_ids_list)
            #     # print(to_pred_phrase[positive_to_pred_phrase_idx_gt_phrase_idx[0]].pred_object_ids_list)
            #     # print(to_pred_phrase[positive_to_pred_phrase_idx_gt_phrase_idx[0]].pred_subject_classes)
            #     # print(to_pred_phrase[positive_to_pred_phrase_idx_gt_phrase_idx[0]].pred_object_classes)
            #     # exit()
            #     positive_topk = min(positive_to_pred_phrase_pred_subject_mask_ios_list_gt_mask.shape[0], 500)
            #     negative_topk = min(negative_to_pred_phrase_pred_subject_mask_ios_list_gt_mask.shape[0], positive_topk)
            #     positive_to_pred_phrase_subobj_crowd_pred_mask_ios_gt_mask_topk, positive_to_pred_phrase_select_by_mask_topk = torch.topk(positive_to_pred_phrase_pred_subject_mask_ios_list_gt_mask + positive_to_pred_phrase_pred_object_mask_ios_list_gt_mask,positive_topk)
            #     negative_to_pred_phrase_subobj_crowd_pred_mask_ios_gt_mask_topk, negative_to_pred_phrase_select_by_mask_topk = torch.topk(negative_to_pred_phrase_pred_subject_mask_ios_list_gt_mask + negative_to_pred_phrase_pred_object_mask_ios_list_gt_mask,negative_topk)
            #     print("positive to pred phrase",positive_to_pred_phrase_subobj_crowd_pred_mask_ios_gt_mask_topk.shape[0])
            #     print("negative to pred phrase",negative_to_pred_phrase_subobj_crowd_pred_mask_ios_gt_mask_topk.shape[0])
            #     exit()
            #     positive_to_pred_phrase=to_pred_phrase[positive_to_pred_phrase_idx_gt_phrase_idx[0]][positive_to_pred_phrase_select_by_mask_topk]
            #     negative_to_pred_phrase=to_pred_phrase[negative_to_pred_phrase_idx_gt_phrase_idx[0]][negative_to_pred_phrase_select_by_mask_topk]
            #     to_pred_phrase=Instances.cat([positive_to_pred_phrase,negative_to_pred_phrase])
            #     to_pred_phrase.gt_relation_onehots=torch.cat([gt_phrase.gt_relation_onehots[positive_to_pred_phrase_idx_gt_phrase_idx[1]][positive_to_pred_phrase_select_by_mask_topk],
            #                                                   torch.cat([torch.ones((negative_to_pred_phrase_subobj_crowd_pred_mask_ios_gt_mask_topk.shape[0],1)),torch.zeros((negative_to_pred_phrase_subobj_crowd_pred_mask_ios_gt_mask_topk.shape[0],self.relation_classes))],dim=1).to(self.device)],dim=0)
            #     negative_subject_in_crowd=torch.zeros_like(negative_to_pred_phrase.pred_subject_ids_list)
            #     negative_subject_in_crowd[negative_to_pred_phrase.pred_subject_ids_list>=0]=1
            #     negative_object_in_crowd=torch.zeros_like(negative_to_pred_phrase.pred_object_ids_list)
            #     negative_object_in_crowd[negative_to_pred_phrase.pred_object_ids_list>=0]=1
            #
            #     to_pred_phrase.pred_subject_gt_in_crowds_list=torch.cat([to_pred_phrase_gt_mask_ios_lists_pred_subject_mask[positive_to_pred_phrase_idx_gt_phrase_idx][positive_to_pred_phrase_select_by_mask_topk],
            #                                                              negative_subject_in_crowd],dim=0)
            #     to_pred_phrase.pred_object_gt_in_crowds_list=torch.cat([to_pred_phrase_gt_mask_ios_lists_pred_object_mask[positive_to_pred_phrase_idx_gt_phrase_idx][positive_to_pred_phrase_select_by_mask_topk],
            #                                                             negative_object_in_crowd],dim=0)
        else:
            return to_pred_crowd, to_pred_phrase
        return to_pred_crowd, to_pred_phrase

# final
@META_ARCH_REGISTRY.register()
class GeneralizedRCNNDistanceSimilarityAllEncodeTrain(GeneralizedRCNN):

    def __init__(self, cfg):
        super(GeneralizedRCNNDistanceSimilarityAllEncodeTrain, self).__init__(cfg)
        self.location_dim=cfg.MODEL.ROI_HEADS.LOCATION_DIM
        self.similarity_feature=cfg.MODEL.ROI_HEADS.SIMILARITY_FEATURE

        # self.location_encode = nn.Linear(self.location_dim, 1024)
        # self.location_ac = nn.ReLU()

        # nn.init.normal_(self.instance_encoder.weight, mean=0, std=0.01)
        # nn.init.constant_(self.instance_encoder.bias, 0)
        # nn.init.normal_(self.location_encode.weight, mean=0, std=0.01)
        # nn.init.constant_(self.location_encode.bias, 0)

        self.to(self.device)

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

    def gt_single_area_generate_to_pred_crowds_for_to_pred_phrases(self,pred_instance,pred_crowd,gt_crowd,training=True,compute_crowd_loss=False):
        # print(pred_instance.image_size)
        # print(pred_instance.pred_boxes)

        to_pred_crowd=Instances(pred_instance.image_size)
        pred_instance_boxes_pad0 = torch.cat([pred_instance.pred_boxes.tensor,torch.Tensor([[pred_instance.image_size[1],pred_instance.image_size[0],0,0]]).to(self.device)])

        class_set=torch.unique(pred_instance.pred_classes)

        to_pred_crowd_raw_crowd_ids=torch.Tensor(range(len(pred_crowd))).to(self.device).unsqueeze(1).repeat(1,class_set.shape[0]).flatten().long()
        to_pred_crowd_classes=class_set.unsqueeze(0).repeat(len(pred_crowd),1).flatten()

        to_pred_crowd_instance_ids_list=torch.Tensor(range(len(pred_instance))).to(self.device).unsqueeze(0).repeat(to_pred_crowd_raw_crowd_ids.shape[0],1).long()
        to_pred_crowd_instance_classes_list=pred_instance.pred_classes.unsqueeze(0).repeat(to_pred_crowd_raw_crowd_ids.shape[0],1)

        to_pred_crowd_boxes=pred_crowd.pred_boxes.tensor[to_pred_crowd_raw_crowd_ids]
        crowd_instance_ios = compute_boxes_io(to_pred_crowd_boxes, pred_instance.pred_boxes.tensor)
        crowd_class_instance_class_hit=to_pred_crowd_instance_classes_list == to_pred_crowd_classes.unsqueeze(1).repeat(1, len(pred_instance))
        matched_crowd_instance = crowd_instance_ios * (crowd_class_instance_class_hit)
        to_pred_crowd_instance_ids_list[matched_crowd_instance<self.positive_box_threshold]=-1
        matched_crowd_instance[matched_crowd_instance<self.positive_box_threshold] = 0
        to_pred_crowd_instance_ids_list, _ = torch.sort(to_pred_crowd_instance_ids_list, dim=1,descending=True)

        unique_to_pred_crowd_instance_ids_list = torch.unique(to_pred_crowd_instance_ids_list,dim=0)
        unique_to_pred_crowd_instance_ids_list = unique_to_pred_crowd_instance_ids_list[torch.sum(unique_to_pred_crowd_instance_ids_list>=0,dim=1)>0]
        to_pred_crowd_pred_classes = pred_instance.pred_classes[unique_to_pred_crowd_instance_ids_list[:, 0]]
        # print(unique_to_pred_crowd_instance_ids_list[0])
        # print(pred_instance_boxes_pad0[unique_to_pred_crowd_instance_ids_list][0])
        # print(union_box_matrix(pred_instance_boxes_pad0[unique_to_pred_crowd_instance_ids_list])[0])
        to_pred_crowd_pred_boxes = union_box_matrix(pred_instance_boxes_pad0[unique_to_pred_crowd_instance_ids_list])
        to_pred_crowd_pred_instance_lens = torch.sum(unique_to_pred_crowd_instance_ids_list >= 0, dim=1)

        max_len=torch.max(to_pred_crowd_pred_instance_lens)
        unique_to_pred_crowd_instance_ids_list=unique_to_pred_crowd_instance_ids_list[:,:max_len]
        to_pred_crowd.pred_instance_ids_list = unique_to_pred_crowd_instance_ids_list.long()
        to_pred_crowd.pred_classes = to_pred_crowd_pred_classes.long()
        to_pred_crowd.pred_instance_lens = to_pred_crowd_pred_instance_lens.long()
        to_pred_crowd.pred_boxes = Boxes(to_pred_crowd_pred_boxes)
        to_pred_crowd.pred_in_clust = torch.ones_like(unique_to_pred_crowd_instance_ids_list)

        if not training:
            _,to_pred_crowd_select=torch.topk(to_pred_crowd_pred_instance_lens,min(min(int(2000/len(pred_instance)),100),to_pred_crowd_pred_instance_lens.shape[0]))
            to_pred_crowd=to_pred_crowd[to_pred_crowd_select]
        # if training:
        #     to_pred_crowd_gt_crowd_class_hit=to_pred_crowd.pred_classes.unsqueeze(1).repeat(1,len(gt_crowd))==gt_crowd.gt_classes.unsqueeze(0).repeat(to_pred_crowd.pred_classes.shape[0],1) # crowd,gt_crowd
        #     pred_ins_ids=to_pred_crowd.pred_instance_ids_list.unsqueeze(1).repeat(1,len(gt_crowd),1).unsqueeze(3).repeat(1,1,1,gt_crowd.gt_instance_ids_list.shape[1]) # crowd,gt_crowd,pred_instance,gt_instance
        #     pred_ins_ids[pred_ins_ids==-1]=-2
        #     gt_ins_ids = gt_crowd.gt_instance_ids_list.unsqueeze(1).repeat(1, len(pred_instance), 1).unsqueeze(0).repeat(to_pred_crowd.pred_instance_ids_list.shape[0],1,1,1) # crowd,gt_crowd,instance,gt_instance
        #     to_pred_crowd_gt_mask_ios_lists_pred_instance_mask = (torch.sum(pred_ins_ids == gt_ins_ids, dim=3) > 0).float() # crowd,gt_crowd,instance
        #     to_pred_crowd_gt_mask_ios_lists_pred_instance_mask=to_pred_crowd_gt_mask_ios_lists_pred_instance_mask * to_pred_crowd_gt_crowd_class_hit.unsqueeze(2)
        #     to_pred_crowd_pred_mask_ios_list_gt_mask=(torch.sum(to_pred_crowd_gt_mask_ios_lists_pred_instance_mask,dim=2).float()/gt_crowd.gt_instance_lens.unsqueeze(0).repeat(to_pred_crowd_gt_mask_ios_lists_pred_instance_mask.shape[0],1)) * to_pred_crowd_gt_crowd_class_hit
        #     to_pred_crowd.pred_mask_ios_list_gt_mask=to_pred_crowd_pred_mask_ios_list_gt_mask
        #     to_pred_crowd.gt_mask_ios_lists_pred_instance_mask=to_pred_crowd_gt_mask_ios_lists_pred_instance_mask
        return to_pred_crowd

    def gt_single_area_generate_to_pred_crowds_for_to_pred_phrases_with_similarity(self,pred_instance,pred_crowd,instance_similarity,training=False):
        instance_similarity=instance_similarity-torch.eye(len(pred_instance)).to(self.device)
        # print(instance_similarity)
        class_set = torch.unique(pred_instance.pred_classes)
        pred_instance_boxes_pad0 = torch.cat([pred_instance.pred_boxes.tensor, torch.Tensor(
            [[pred_instance.image_size[1], pred_instance.image_size[0], 0, 0]]).to(self.device)])
        left_instance_classes = pred_instance.pred_classes.unsqueeze(1).repeat(1, len(pred_instance))
        right_instance_classes = pred_instance.pred_classes.unsqueeze(0).repeat(len(pred_instance), 1)
        sameclass = (left_instance_classes == right_instance_classes).int()
        sameclass[sameclass == 0] = -1
        class_matrix = (left_instance_classes + 1) * (sameclass)
        class_matrix[class_matrix < 0] = -1
        class_matrix = class_matrix - 1

        to_pred_crowd_instance_ids_list = []
        # print(pred_instance)
        max_len=1
        for thr in self.similarity:#[0.8,0.5,0.25,0.05,0]:
            # print(thr)
            # print((instance_similarity >= thr).int())
            for ins_cls in class_set:
                cls_instances=(class_matrix == torch.ones(len(pred_instance), len(pred_instance)).to(self.device) * ins_cls).int()
                # if thr==0.5:
                #     print(torch.where(cls_instances==1))
                # print(ins_cls)
                # print(instance_similarity*cls_instances)
                crowd_matrix=(instance_similarity >= thr)*cls_instances
                crowd=self.components(crowd_matrix)
                # crowd=torch.unique(torch.where(crowd_matrix==1)[0])
                if len(crowd)>0:
                    # print(crowd)
                    to_pred_crowd_instance_ids_list.extend(crowd)
                    for c in crowd:
                        max_len=max(max_len,len(c))
        # print(to_pred_crowd_instance_ids_list)
        # exit()
        exist_instance_ids=[]
        for i in range(len(to_pred_crowd_instance_ids_list)):
            to_pred_crowd_instance_ids_list[i]=torch.cat([to_pred_crowd_instance_ids_list[i],torch.ones(max_len-len(to_pred_crowd_instance_ids_list[i])).to(self.device)*(-1)])
            exist_instance_ids.append(to_pred_crowd_instance_ids_list[i])
        if len(exist_instance_ids)>0:
            exist_instance_ids=torch.unique(torch.cat(exist_instance_ids))
            noexist=set(range(0,len(pred_instance))).difference(set(exist_instance_ids.data.cpu().numpy().tolist()))
        else:
            noexist=set(range(0,len(pred_instance)))
        for i in noexist:
            instance_id_tensor=torch.Tensor([i]).to(self.device)
            none_instance_ids_tensor=torch.ones(max_len - 1).to(self.device) * (-1)
            to_pred_crowd_instance_ids_list.append(torch.cat([instance_id_tensor,none_instance_ids_tensor]))
        # print(to_pred_crowd_instance_ids_list)
        to_pred_crowd_instance_ids_list=torch.stack(to_pred_crowd_instance_ids_list).long()
        to_pred_crowd_instance_ids_list=torch.unique(to_pred_crowd_instance_ids_list,dim=0)
        # print(to_pred_crowd_instance_ids_list)
        # exit()
        to_pred_crowd=Instances(pred_instance.image_size)
        to_pred_crowd.pred_instance_ids_list=torch.sort(to_pred_crowd_instance_ids_list,dim=1,descending=True)[0]
        to_pred_crowd.pred_instance_lens=torch.sum(to_pred_crowd.pred_instance_ids_list>=0,dim=1)
        to_pred_crowd.pred_boxes=Boxes(union_box_matrix(pred_instance_boxes_pad0[to_pred_crowd.pred_instance_ids_list]))
        to_pred_crowd.pred_classes=pred_instance.pred_classes[to_pred_crowd.pred_instance_ids_list[:,0]].long()
        to_pred_crowd.pred_in_clust=torch.ones_like(to_pred_crowd.pred_instance_ids_list)
        return to_pred_crowd

    def forward(self, batched_inputs,iteration, mode, training, print_size=False):
        self.print_size = print_size
        losses = {}
        results=[]
        metrics={}

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if mode=="relation":
            relation_results=None

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            if "crowds" in batched_inputs[0]:
                gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
            else:
                gt_crowds = None
            if "phrases" in batched_inputs[0]:
                gt_phrases = [x["phrases"].to(self.device) for x in batched_inputs]
            else:
                gt_phrases = None
            if self.use_gt_instance:
                for image_i in range(len(gt_instances)):
                    gt_instances[image_i].pred_classes = gt_instances[image_i].gt_classes
                    gt_instances[image_i].pred_boxes = gt_instances[image_i].gt_boxes
                    gt_instances[image_i].pred_full_masks = gt_instances[image_i].gt_masks.tensor
                instance_detections=gt_instances
                instance_local_features_roi, instance_local_features = self.roi_heads.generate_instance_box_features(features,instance_detections)
            else:
                if self.proposal_generator:
                    instance_proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images,features,None,False)
                else:
                    assert "proposals" in batched_inputs[0]
                    instance_proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                instance_detections, instance_local_features_roi, instance_local_features, instance_detector_loss = self.roi_heads(images, features,instance_proposals, None,"instance", False)
                self.fullize_mask(instance_detections)
            instance_location_features=[]
            for pred_instance in instance_detections:
                global_location = torch.Tensor([0, 0, pred_instance.image_size[1], pred_instance.image_size[0]]).to(self.device)
                global_location = global_location.unsqueeze(0).repeat(len(pred_instance), 1)
                instance_location_feature = pred_instance.pred_boxes.tensor - global_location  # x1,y1,x2,y2
                global_location = torch.Tensor([pred_instance.image_size[1], pred_instance.image_size[0], pred_instance.image_size[1],pred_instance.image_size[0]]).to(self.device)
                global_location = global_location.unsqueeze(0).repeat(len(pred_instance), 1)
                instance_location_feature = instance_location_feature / global_location
                instance_location_features.append(instance_location_feature)

            if self.instance_encoder:
                instance_local_features_encode = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
            else:
                instance_local_features_encode=instance_local_features
            # instance_local_features_pad0=[]

            instance_similarities=[]
            for pred_instance,instance_local_feature,instance_location_feature in zip(instance_detections,instance_local_features_encode,instance_location_features):
                class_similarity_score = (pred_instance.pred_classes.unsqueeze(1).repeat(1, len(pred_instance)) == pred_instance.pred_classes.unsqueeze(0).repeat(len(pred_instance),1)).float()

                if self.similarity_feature=="loc":
                    instance_feature=instance_location_feature
                elif self.similarity_feature=="vis":
                    instance_feature = instance_local_feature
                elif self.similarity_feature=="visloc":
                    instance_feature = torch.cat([instance_local_feature,instance_location_feature],dim=-1)
                else:
                    print("NO SUCH SIMILARITY FEATURE",self.similarity_feature)
                    exit()

                if self.similarity_mode == "cosine":
                    simi_flatten = F.cosine_similarity(
                        instance_feature.unsqueeze(1).repeat(1, instance_feature.shape[0], 1).view(
                            instance_feature.shape[0] * instance_feature.shape[0], instance_feature.shape[1]),
                        instance_feature.unsqueeze(0).repeat(instance_feature.shape[0], 1, 1).view(
                            instance_feature.shape[0] * instance_feature.shape[0], instance_feature.shape[1]))
                    instance_similarity = simi_flatten.view(instance_feature.shape[0],instance_feature.shape[0])
                elif self.similarity_mode == "distance":
                    instance_distance = calc_self_distance(instance_feature)
                    instance_distance = instance_distance.view(len(pred_instance), len(pred_instance))
                    instance_similarity = 1 - instance_distance

                if self.gt_similarity_mode=="support_normal_class":
                    instance_groups = []
                    unique_classes = torch.unique(pred_instance.pred_classes)
                    pred_class_similarity=torch.zeros_like(instance_similarity)
                    for cls in unique_classes:
                        instance_group = torch.where(pred_instance.pred_classes == cls)[0]
                        instance_groups.append(instance_group)
                    for instance_group in instance_groups:
                        instance_group_min = 2e15
                        for instance_idx1 in instance_group:
                            for instance_idx2 in instance_group:
                                instance_group_min = min(instance_similarity[instance_idx1][instance_idx2],instance_group_min)
                        if instance_group_min==1:
                            instance_group_min=instance_group_min-0.00001
                        for instance_idx1 in instance_group:
                            for instance_idx2 in instance_group:
                                pred_class_similarity[instance_idx1][instance_idx2]=(instance_similarity[instance_idx1][instance_idx2]-instance_group_min)/(1-instance_group_min)
                    instance_similarities.append(pred_class_similarity)
                else:
                    instance_similarities.append((instance_similarity - torch.min(instance_similarity.flatten())) /
                                                 (1 - torch.min(instance_similarity.flatten())))

            # if training:
                # gt_instance_similarities=self.gt_similarity(instance_detections,gt_instances,gt_crowds)
                # print(instance_detections[0].pred_classes)
                # print(class_similarity_score)
                # print(gt_instance_similarities[0])
            # print(instance_similarity)
            # print((instance_similarity - torch.min(instance_similarity.flatten())) / (1 - torch.min(instance_similarity.flatten()))*class_similarity_score)
            # print(instance_similarities[0]*class_similarity_score)
            # exit()
                # exit()
            if self.print_size:
                if gt_instances:
                    print("pred instances",len(instance_detections[0]),"gt instances",len(gt_instances[0]))
                else:
                    print("pred instances", len(instance_detections[0]))

            error_happen=False
            for instance_detection in instance_detections:
                if len(instance_detection)<=0:
                    error_happen=True
                    break
            if not error_happen:
                crowd_proposals=[]
                for instance_detection in instance_detections:
                    crowd_proposal=Instances(instance_detection.image_size)
                    crowd_proposal.pred_boxes = Boxes(torch.Tensor([[0, 0, instance_detection.image_size[1], instance_detection.image_size[0]]]).to(self.device))
                    crowd_proposals.append(crowd_proposal)

                to_pred_crowds = []
                for image_i in range(len(instance_detections)):
                    # if training:
                    #     to_pred_crowd = self.gt_single_area_generate_to_pred_crowds_for_to_pred_phrases(
                    #         instance_detections[image_i], crowd_proposals[image_i], gt_crowds[image_i], training)
                    # else:
                    to_pred_crowd=self.gt_single_area_generate_to_pred_crowds_for_to_pred_phrases_with_similarity(
                        instance_detections[image_i], crowd_proposals[image_i], instance_similarities[image_i], training
                    )
                    # print(instance_detections[0].pred_classes)
                    # print(instance_similarities[0])
                    # print(to_pred_crowd.pred_classes)
                    # print(to_pred_crowd.pred_instance_ids_list)
                    # print("-------------------")
                    to_pred_crowds.append(to_pred_crowd)

                to_pred_crowds, to_pred_phrases = self.generate_to_pred_phrases_from_to_pred_crowds(instance_detections, to_pred_crowds,gt_crowds, gt_phrases, training)
                # print(to_pred_crowds[0].pred_instance_ids_list)
                if self.print_size:
                    print("to_pred_crowds", len(to_pred_crowds[0]))
                    if gt_phrases:
                        print("to_pred_phrases", len(to_pred_phrases[0]), "gt phrases", len(gt_phrases[0]))
                    else:
                        print("to_pred_phrases", len(to_pred_phrases[0]))
                instance_detections[0].remove("pred_full_masks")

                if len(to_pred_phrases[0]) > 0:
                    if len(to_pred_phrases[0]) > 0:
                        to_pred_crowd_local_features_encode = None
                        to_pred_phrase_local_features_encode = None
                        if "crowd" in self.relation_head_list:
                            to_pred_crowd_local_features_roi, to_pred_crowd_local_features = self.roi_heads.generate_instance_box_features(
                                features, to_pred_crowds)
                            if self.instance_encoder:
                                to_pred_crowd_local_features_encode = [
                                    self.instance_encoder(to_pred_crowd_local_feature) for to_pred_crowd_local_feature
                                    in to_pred_crowd_local_features]
                            else:
                                to_pred_crowd_local_features_encode = to_pred_crowd_local_features
                        if "phrase" in self.relation_head_list:

                            to_pred_phrase_local_features_roi, to_pred_phrase_local_features = self.roi_heads.generate_instance_box_features(
                                features, to_pred_phrases)
                            if self.instance_encoder:
                                to_pred_phrase_local_features_encode = [
                                    self.instance_encoder(to_pred_phrase_local_feature) for to_pred_phrase_local_feature
                                    in to_pred_phrase_local_features]
                            else:
                                to_pred_phrase_local_features_encode = to_pred_phrase_local_features
                        relation_results, rela_losses_reduced, rela_metrics = self.relation_heads(instance_detections,
                                                                                                  instance_local_features_encode,
                                                                                                  to_pred_crowds,
                                                                                                  to_pred_crowd_local_features_encode,
                                                                                                  None,
                                                                                                  to_pred_phrases,
                                                                                                  to_pred_phrase_local_features_encode,
                                                                                                  training=training,
                                                                                                  iteration=iteration)
                    # print(relation_results[0].pred_subject_ids_list)
                    # print(relation_results[0].pred_subject_in_crowds_list)
                    # print("--------------------------")
                    losses.update(rela_losses_reduced)
                else:
                    losses = {}
            else:
                losses={}

            if not training:
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes, "instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    results[i].update(instance_results[i])
                    if relation_results:
                        results[i]['phrases']=relation_results[i]

        return results, losses, metrics

    def generate_crowd_instance_dict(self,pred_instances):
        crowd_instance_dict_list=[]
        max_len_list=[]
        for image_i in range(len(pred_instances)):
            pred_instance = pred_instances[image_i]
            instance_dict = defaultdict(list)
            max_len = 0
            for instance_i in range(len(pred_instance)):
                cls = str(pred_instance.pred_classes[instance_i].item())
                instance_dict[cls].append(torch.Tensor([instance_i]).squeeze(0).to(self.device).long())
                max_len = max(max_len, len(instance_dict[cls]))
            max_len_list.append(max_len)
            crowd_instance_dict_list.append(instance_dict)
        return crowd_instance_dict_list,max_len_list

@META_ARCH_REGISTRY.register()
class GeneralizedRCNNWithCrowdProposal(GeneralizedRCNN):

    def __init__(self, cfg):
        super(GeneralizedRCNNWithCrowdProposal, self).__init__(cfg)
        self.crowd_proposal_generator = build_crowd_proposal_generator(cfg, self.backbone.output_shape(),type="crowd")
        self.to(self.device)

    def forward(self, batched_inputs, iteration, mode, training, print_size=False):  # batched_imputs['height'] and batched_imputs['width'] is real
        self.print_size = print_size
        losses = {}
        results=[]
        metrics={}

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if mode=="crowd":
            if training:
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                else:
                    gt_instances = None
                for image_i in range(len(gt_instances)):
                    gt_instances[image_i].pred_classes = gt_instances[image_i].gt_classes
                    gt_instances[image_i].pred_boxes = gt_instances[image_i].gt_boxes
                    gt_instances[image_i].pred_full_masks = gt_instances[image_i].gt_masks.tensor
                # if self.proposal_generator:
                #     proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                # else:
                #     print("need proposal generator")
                #     exit()
                # instance_detections, instance_local_features, instance_detector_loss = self.roi_heads(images, features, proposals, None,"instance",False)
                # self.fullize_mask(instance_detections)
                # if self.instance_encoder:
                #     instance_local_features = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
                # print("instance",len(instance_detections[0]),"gt instance",len(gt_instances[0]))

                if "crowds" in batched_inputs[0]:
                    gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
                else:
                    gt_crowds = None
                if self.crowd_proposal_generator:
                    crowd_proposals, crowd_proposal_losses, crowd_ctrness_pred, crowd_global_features = self.crowd_proposal_generator(images, features, gt_crowds, True)
                else:
                    assert "crowd_proposals" in batched_inputs[0]
                    crowd_proposals = [x["crowd_proposals"].to(self.device) for x in batched_inputs]
                    crowd_proposal_losses = {}
                losses.update(crowd_proposal_losses)
                # if crowd_proposals[0].has("proposal_boxes"):
                #     for image_i in range(len(crowd_proposals)):
                #         crowd_proposals[image_i].pred_boxes = crowd_proposals[image_i].proposal_boxes
                # pred_crowd_pred_instance_dict_list, max_len_list = self.generate_crowd_instance_dict(gt_instances, crowd_proposals)
                # to_pred_crowds = []
                # crowd_match_losses = []
                # for image_i in range(len(gt_instances)):
                #     to_pred_crowd, crowd_match_loss = self.single_area_generate_to_pred_crowds_for_to_pred_phrases(gt_instances[image_i], pred_crowd_pred_instance_dict_list[image_i], max_len_list[image_i], gt_crowds[image_i], training,compute_crowd_loss=True,pred_crowd=crowd_proposals[image_i])
                #     to_pred_crowds.append(to_pred_crowd)
                #     crowd_match_losses.append(crowd_match_loss)
                # losses['loss_crowd_match']=torch.mean(torch.stack(crowd_match_losses))

                # if gt_crowds:
                #     print("crowd proposal", len(crowd_proposals[0]), "to_pred_crowds", len(to_pred_crowds[0]),"gt crowds", len(gt_crowds[0]))
                # else:
                #     print("crowd proposal", len(crowd_proposals[0]), "to_pred_crowds", len(to_pred_crowds[0]))
            else:
                if self.proposal_generator:
                    proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                else:
                    print("need proposal generator")
                    exit()
                instance_detections, instance_local_features, instance_detector_loss = self.roi_heads(images, features, proposals, None,"instance",False)
                if self.instance_encoder:
                    instance_local_features = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
                print("instance", len(instance_detections[0]))

                if len(instance_detections[0])>0:
                    if self.crowd_proposal_generator:
                        crowd_proposals, crowd_proposal_losses, crowd_ctrness_pred, crowd_global_features = self.crowd_proposal_generator(images, features, None, training)
                    else:
                        assert "crowd_proposals" in batched_inputs[0]
                        crowd_proposals = [x["crowd_proposals"].to(self.device) for x in batched_inputs]
                        crowd_proposal_losses = {}
                    if crowd_proposals[0].has("proposal_boxes"):
                        for image_i in range(len(crowd_proposals)):
                            crowd_proposals[image_i].pred_boxes = crowd_proposals[image_i].proposal_boxes
                    crowd_proposals=self.filter_small_proposals(instance_detections,crowd_proposals)
                    print("crowd proposals", len(crowd_proposals[0]))
                    crowd_results = self._postprocess(crowd_proposals, batched_inputs, images.image_sizes,"crowd")
                    for i, r in enumerate(crowd_results):
                        if i >= len(results):
                            results.append({})
                        results[i].update(crowd_results[i])

        if mode=="relation":
            relation_results=None

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            if "crowds" in batched_inputs[0]:
                gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
            else:
                gt_crowds = None
            if "phrases" in batched_inputs[0]:
                gt_phrases = [x["phrases"].to(self.device) for x in batched_inputs]
            else:
                gt_phrases = None

            if self.use_gt_instance:
                for image_i in range(len(gt_instances)):
                    gt_instances[image_i].pred_classes = gt_instances[image_i].gt_classes
                    gt_instances[image_i].pred_boxes = gt_instances[image_i].gt_boxes
                    gt_instances[image_i].pred_full_masks = gt_instances[image_i].gt_masks.tensor
                instance_detections=gt_instances
                instance_local_features = self.roi_heads.generate_instance_box_features(features, instance_detections)
            else:
                if self.proposal_generator:
                    proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                else:
                    print("need proposal generator")
                    exit()
                instance_detections, instance_local_features, instance_detector_loss = self.roi_heads(images, features, proposals, None,"instance",False)
                self.fullize_mask(instance_detections)
                if training and (not self.use_gt_instance):
                    self.replace_gt_instance_with_pred_instances(instance_detections,gt_instances,gt_crowds,gt_phrases)
            if self.instance_encoder:
                instance_local_features = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
            if self.print_size:
                if gt_instances:
                    print("instance",len(instance_detections[0]),"gt instance",len(gt_instances[0]))
                    print(instance_detections[0].pred_classes)
                else:
                    print("instance", len(instance_detections[0]))

            if len(instance_detections[0])>0:
                if self.crowd_proposal_generator:
                    crowd_proposals, crowd_proposal_losses, crowd_ctrness_pred, crowd_global_features = self.crowd_proposal_generator(images, features, gt_crowds, training)
                else:
                    assert "crowd_proposals" in batched_inputs[0]
                    crowd_proposals = [x["crowd_proposals"].to(self.device) for x in batched_inputs]
                    crowd_proposal_losses = {}
                losses.update(crowd_proposal_losses)
                if crowd_proposals[0].has("proposal_boxes"):
                    for image_i in range(len(crowd_proposals)):
                        crowd_proposals[image_i].pred_boxes = crowd_proposals[image_i].proposal_boxes
                crowd_proposals = self.filter_small_proposals(instance_detections, crowd_proposals)

                crowd_proposals_all=Instances(instance_detections[0].image_size)
                crowd_proposals_all.pred_boxes=Boxes(torch.cat([crowd_proposals[0].pred_boxes.tensor,torch.Tensor([[0,0,instance_detections[0].image_size[1],instance_detections[0].image_size[0]]]).to(self.device)]))
                # crowd_proposals_all.pred_boxes = Boxes(torch.Tensor([[0, 0, instance_detections[0].image_size[1], instance_detections[0].image_size[0]]]).to(self.device))
                crowd_proposals[0]=crowd_proposals_all
                if False:
                    pred_crowd_pred_instance_dict_list,max_len_list=self.generate_crowd_instance_dict(instance_detections, crowd_proposals)
                to_pred_crowds=[]
                for image_i in range(len(instance_detections)):
                    if False:
                        to_pred_crowd = self.single_area_generate_to_pred_crowds_for_to_pred_phrases(instance_detections[image_i], pred_crowd_pred_instance_dict_list[image_i],max_len_list[image_i], gt_crowds[image_i], training)
                    if training:
                        to_pred_crowd = self.gt_single_area_generate_to_pred_crowds_for_to_pred_phrases(instance_detections[image_i], crowd_proposals[image_i], gt_crowds[image_i],training)
                    else:
                        to_pred_crowd = self.gt_single_area_generate_to_pred_crowds_for_to_pred_phrases(instance_detections[image_i], crowd_proposals[image_i], None,training)

                    to_pred_crowds.append(to_pred_crowd)
                if self.print_size:
                    if gt_crowds:
                        print("crowd proposal", len(crowd_proposals[0]), "to_pred_crowds", len(to_pred_crowds[0]), "gt crowds", len(gt_crowds[0]))
                    else:
                        print("crowd proposal", len(crowd_proposals[0]),"to_pred_crowds", len(to_pred_crowds[0]))

                to_pred_crowds, to_pred_phrases = self.generate_to_pred_phrases_from_to_pred_crowds(instance_detections, to_pred_crowds, gt_crowds, gt_phrases,training)
                if self.print_size:
                    if gt_phrases:
                        print("to_pred_phrases", len(to_pred_phrases[0]),"gt phrases", len(gt_phrases[0]))
                    else:
                        print("to_pred_phrases", len(to_pred_phrases[0]))

                instance_detections[0].remove("pred_full_masks")

                if len(to_pred_phrases[0])>0:
                    to_pred_phrase_local_features=None
                    crowd_local_features = None
                    if "crowd" in self.relation_head_list:
                        crowd_local_features = self.roi_heads.generate_instance_box_features(features,to_pred_crowds)

                    if "phrase" in self.relation_head_list:
                        to_pred_phrase_local_features = self.roi_heads.generate_instance_box_features(features, to_pred_phrases)
                    relation_results, rela_losses, rela_metrics = self.relation_heads(instance_detections,instance_local_features,
                                                                                      to_pred_phrases, to_pred_phrase_local_features,
                                                                                      to_pred_crowds,crowd_local_features,None,None,training=training,iteration=iteration)

                    losses.update(rela_losses)
                else:
                    losses = {}
            else:
                losses={}

            if not training:
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes,"instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    if relation_results:
                        results[i]['phrases']=relation_results[i]
                    results[i].update(instance_results[i])

        return results, losses, metrics

    def generate_crowd_instance_dict(self, pred_instances, pred_crowds):
        crowd_instance_dict_list=[]
        max_len_list=[]
        for image_i in range(len(pred_crowds)):
            pred_instance=pred_instances[image_i]
            pred_crowd=pred_crowds[image_i]

            crowd_instance_io = compute_boxes_io(pred_crowd.pred_boxes.tensor, pred_instance.pred_boxes.tensor)
            crowd_ids, instance_ids = torch.where(crowd_instance_io > 0.5)
            instance_classes = pred_instance.pred_classes[instance_ids]

            max_len = 0
            crowd_instance=defaultdict(list)
            for crowd_id, instance_id, instance_class in zip(crowd_ids, instance_ids, instance_classes):
                key = str(crowd_id.item()) + "_" + str(instance_class.item())
                crowd_instance[key].append(instance_id)
                max_len = max(max_len, len(crowd_instance[key]))

            crowd_instance_dict_list.append(crowd_instance)
            max_len_list.append(max_len)
        return crowd_instance_dict_list,max_len_list

@META_ARCH_REGISTRY.register()
class GeneralizedRCNNWithPhraseProposal(GeneralizedRCNN):

    def __init__(self, cfg):
        super(GeneralizedRCNNWithPhraseProposal, self).__init__(cfg)
        self.phrase_proposal_generator = build_phrase_proposal_generator(cfg, self.backbone.output_shape(),type="phrase")
        self.to(self.device)

    def forward(self, batched_inputs, iteration, mode, training, print_size=False):  # batched_imputs['height'] and batched_imputs['width'] is real
        self.print_size = print_size
        losses = {}
        results=[]
        metrics={}

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if mode=="phrase":
            if training:
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                else:
                    gt_instances = None
                for image_i in range(len(gt_instances)):
                    gt_instances[image_i].pred_classes = gt_instances[image_i].gt_classes
                    gt_instances[image_i].pred_boxes = gt_instances[image_i].gt_boxes
                    gt_instances[image_i].pred_full_masks = gt_instances[image_i].gt_masks.tensor
                # if self.proposal_generator:
                #     proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                # else:
                #     print("need proposal generator")
                #     exit()
                # instance_detections, instance_local_features, instance_detector_loss = self.roi_heads(images, features, proposals, None,"instance",False)
                # if self.instance_encoder:
                #     instance_local_features=self.instance_encoder(instance_local_features)
                # self.fullize_mask(instance_detections)
                # print("instance",len(instance_detections[0]),"gt instance",len(gt_instances[0]))
                if "crowds" in batched_inputs[0]:
                    gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
                else:
                    gt_crowds = None
                if "phrases" in batched_inputs[0]:
                    gt_phrases = [x["phrases"].to(self.device) for x in batched_inputs]
                else:
                    gt_phrases = None
                if self.phrase_proposal_generator:
                    phrase_proposals, phrase_proposal_losses, phrase_ctrness_pred, phrase_global_features = self.phrase_proposal_generator(images, features, gt_phrases, True)
                else:
                    assert "phrase_proposals" in batched_inputs[0]
                    phrase_proposals = [x["phrase_proposals"].to(self.device) for x in batched_inputs]
                    phrase_proposal_losses = {}
                losses.update(phrase_proposal_losses)
                # if phrase_proposals[0].has("proposal_boxes"):
                #     for image_i in range(len(phrase_proposals)):
                #         phrase_proposals[image_i].pred_boxes = phrase_proposals[image_i].proposal_boxes
                # pred_phrase_pred_instance_dict_list, max_len_list = self.generate_phrase_instance_dict(gt_instances,phrase_proposals)
                # to_pred_crowds = []
                # to_pred_phrases = []
                # crowd_match_losses = []
                # for image_i in range(len(gt_instances)):
                #     image_to_pred_crowds = []
                #     image_to_pred_phrases = []
                #     image_to_pred_crowd_num = 0
                #     pred_phrase_pred_instance_dict = pred_phrase_pred_instance_dict_list[image_i]
                #     for phrase_id in pred_phrase_pred_instance_dict:
                #         phrase_to_pred_crowd, crowd_match_loss = self.single_area_generate_to_pred_crowds_for_to_pred_phrases(gt_instances[image_i], pred_phrase_pred_instance_dict[phrase_id], max_len_list[image_i],gt_crowds[image_i], training,compute_crowd_loss=True,pred_crowd=phrase_proposals[image_i])
                #         crowd_match_losses.append(crowd_match_loss)
                #         phrase_to_pred_phrase = self.single_area_generate_to_pred_phrases_from_to_pred_crowds(gt_instances[image_i], phrase_to_pred_crowd, gt_crowds[image_i], gt_phrases[image_i])
                #         phrase_to_pred_phrase.pred_subject_crowd_ids = phrase_to_pred_phrase.pred_subject_crowd_ids + image_to_pred_crowd_num
                #         phrase_to_pred_phrase.pred_object_crowd_ids = phrase_to_pred_phrase.pred_object_crowd_ids + image_to_pred_crowd_num
                #         image_to_pred_crowds.append(phrase_to_pred_crowd)
                #         image_to_pred_phrases.append(phrase_to_pred_phrase)
                #         image_to_pred_crowd_num += len(phrase_to_pred_crowd)
                #     to_pred_crowds.append(Instances.cat(image_to_pred_crowds))
                #     to_pred_phrases.append(Instances.cat(image_to_pred_phrases))
                # losses['loss_crowd_match'] = torch.mean(torch.stack(crowd_match_losses))
                # if gt_crowds:
                #     print("to_pred_crowds", len(to_pred_crowds[0]), "gt crowd", len(gt_crowds[0]))
                # else:
                #     print("to_pred_crowds", len(to_pred_crowds[0]))
                # print("to_pred_phrases", len(to_pred_phrases[0]))

            else:
                if self.proposal_generator:
                    proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                else:
                    print("need proposal generator")
                    exit()
                instance_detections, instance_local_features, instance_detector_loss = self.roi_heads(images, features, proposals, None,"instance",False)
                if self.instance_encoder:
                    instance_local_features = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
                print("instance", len(instance_detections[0]))

                if len(instance_detections[0])>0:
                    if self.phrase_proposal_generator:
                        phrase_proposals, phrase_proposal_losses, phrase_ctrness_pred, phrase_global_features = self.phrase_proposal_generator(images, features, None, training)
                    else:
                        assert "phrase_proposals" in batched_inputs[0]
                        phrase_proposals = [x["phrase_proposals"].to(self.device) for x in batched_inputs]
                        phrase_proposal_losses = {}
                    if phrase_proposals[0].has("proposal_boxes"):
                        for image_i in range(len(phrase_proposals)):
                            phrase_proposals[image_i].pred_boxes = phrase_proposals[image_i].proposal_boxes
                    phrase_proposals=self.filter_small_proposals(instance_detections,phrase_proposals)
                    print("phrase proposals", len(phrase_proposals[0]))
                    phrase_results = self._postprocess(phrase_proposals, batched_inputs, images.image_sizes,"phrase")
                    for i, r in enumerate(phrase_results):
                        if i >= len(results):
                            results.append({})
                        results[i].update(phrase_results[i])

        if mode=="relation":
            relation_results=None

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            if "crowds" in batched_inputs[0]:
                gt_crowds = [x["crowds"].to(self.device) for x in batched_inputs]
            else:
                gt_crowds = None
            if "phrases" in batched_inputs[0]:
                gt_phrases = [x["phrases"].to(self.device) for x in batched_inputs]
            else:
                gt_phrases = None
            if self.use_gt_instance:
                for image_i in range(len(gt_instances)):
                    gt_instances[image_i].pred_classes = gt_instances[image_i].gt_classes
                    gt_instances[image_i].pred_boxes = gt_instances[image_i].gt_boxes
                    gt_instances[image_i].pred_full_masks = gt_instances[image_i].gt_masks.tensor
                instance_detections=gt_instances
                instance_local_features = self.roi_heads.generate_instance_box_features(features, instance_detections)
            else:
                if self.proposal_generator:
                    instance_proposals, _, instance_ctrness_pred, instance_global_features = self.proposal_generator(images, features, None,False)
                else:
                    print("need proposal generator")
                    exit()
                instance_detections, instance_local_features, instance_detector_loss = self.roi_heads(images, features, instance_proposals, None,"instance",False)
                self.fullize_mask(instance_detections)
                if training and not (self.use_gt_instance):
                    self.replace_gt_instance_with_pred_instances(instance_detections,gt_instances,gt_crowds,gt_phrases)
            if self.instance_encoder:
                instance_local_features = [self.instance_encoder(instance_local_feature) for instance_local_feature in instance_local_features]
            if self.print_size:
                if gt_instances:
                    print("instance",len(instance_detections[0]),"gt instance",len(gt_instances[0]))
                    # print(instance_detections[0].pred_classes)
                else:
                    print("instance", len(instance_detections[0]))

            if len(instance_detections[0])>0:
                if self.phrase_proposal_generator:
                    phrase_proposals, phrase_proposal_losses, phrase_ctrness_pred, phrase_global_features = self.phrase_proposal_generator(images, features, gt_phrases, training)
                else:
                    assert "phrase_proposals" in batched_inputs[0]
                    phrase_proposals = [x["phrase_proposals"].to(self.device) for x in batched_inputs]
                    phrase_proposal_losses = {}
                losses.update(phrase_proposal_losses)
                if phrase_proposals[0].has("proposal_boxes"):
                    for image_i in range(len(phrase_proposals)):
                        phrase_proposals[image_i].pred_boxes = phrase_proposals[image_i].proposal_boxes
                phrase_proposals = self.filter_small_proposals(instance_detections, phrase_proposals)
                phrase_proposals_all=Instances(instance_detections[0].image_size)
                phrase_proposals_all.pred_boxes=Boxes(torch.cat([phrase_proposals[0].pred_boxes.tensor,torch.Tensor([[0,0,instance_detections[0].image_size[1],instance_detections[0].image_size[0]]]).to(self.device)]))
                # phrase_proposals_all.pred_boxes=Boxes(torch.Tensor([[0,0,instance_detections[0].image_size[1],instance_detections[0].image_size[0]]]).to(self.device))
                phrase_proposals[0]=phrase_proposals_all
                if self.print_size:
                    if gt_phrases:
                        print("phrase proposal", len(phrase_proposals[0]),"gt phrase",len(gt_phrases[0]))
                    else:
                        print("phrase proposal", len(phrase_proposals[0]))

                if False:
                    pred_phrase_pred_instance_dict_list, max_len_list = self.generate_phrase_instance_dict(instance_detections, phrase_proposals)
                    to_pred_crowds=[]
                    to_pred_phrases=[]
                    for image_i in range(len(gt_instances)):
                        image_to_pred_crowds = []
                        image_to_pred_phrases = []
                        image_to_pred_crowd_num=0
                        pred_phrase_pred_instance_dict = pred_phrase_pred_instance_dict_list[image_i]
                        for phrase_id in pred_phrase_pred_instance_dict:
                            phrase_to_pred_crowd = self.single_area_generate_to_pred_crowds_for_to_pred_phrases(instance_detections[image_i],pred_phrase_pred_instance_dict[phrase_id],max_len_list[image_i],gt_crowds[image_i],training)
                            phrase_to_pred_phrase = self.single_area_generate_to_pred_phrases_from_to_pred_crowds(instance_detections[image_i],phrase_to_pred_crowd,gt_crowds[image_i],gt_phrases[image_i],training)
                            phrase_to_pred_phrase.pred_subject_crowd_ids=phrase_to_pred_phrase.pred_subject_crowd_ids+image_to_pred_crowd_num
                            phrase_to_pred_phrase.pred_object_crowd_ids = phrase_to_pred_phrase.pred_object_crowd_ids + image_to_pred_crowd_num
                            image_to_pred_crowds.append(phrase_to_pred_crowd)
                            image_to_pred_phrases.append(phrase_to_pred_phrase)
                            image_to_pred_crowd_num+=len(phrase_to_pred_crowd)
                    to_pred_crowds.append(Instances.cat(image_to_pred_crowds))
                    image_to_pred_phrase=Instances.cat(image_to_pred_phrases)
                    _,select_topk=torch.topk(image_to_pred_phrase.to_pred_phrase_subobj_crowd_pred_mask_ios_gt_mask_topk,min(1000,len(image_to_pred_phrase)))
                    to_pred_phrases.append(image_to_pred_phrase[select_topk])
                else:
                    to_pred_crowds=[]
                    to_pred_phrases=[]
                    for image_i in range(len(instance_detections)):
                        if training:
                            to_pred_crowd, to_pred_phrase=self.gt_single_area_generate_to_pred_phrases(instance_detections[image_i],phrase_proposals[image_i],gt_phrases[image_i],training)
                        else:
                            to_pred_crowd, to_pred_phrase = self.gt_single_area_generate_to_pred_phrases(instance_detections[image_i],phrase_proposals[image_i],None, training)
                        to_pred_crowds.append(to_pred_crowd)
                        to_pred_phrases.append(to_pred_phrase)

                if self.print_size:
                    if len(to_pred_crowds)>0:
                        if gt_crowds:
                            print("to_pred_crowds",len(to_pred_crowds[0]),"gt crowd",len(gt_crowds[0]))
                        else:
                            print("to_pred_crowds", len(to_pred_crowds[0]))
                    if len(to_pred_phrases)>0:
                        if gt_phrases:
                            print("to_pred_phrases",len(to_pred_phrases[0]),"gt phrase",len(gt_phrases[0]))
                        else:
                            print("to_pred_phrases", len(to_pred_phrases[0]))

                instance_detections[0].remove("pred_full_masks")

                if len(to_pred_phrases[0])>0:
                    crowd_local_features=None
                    to_pred_phrase_local_features=None
                    if "crowd" in self.relation_head_list:
                        crowd_local_features = self.roi_heads.generate_instance_box_features(features, to_pred_crowds)
                    if "phrase" in self.relation_head_list:
                        to_pred_phrase_local_features = self.roi_heads.generate_instance_box_features(features, to_pred_phrases)

                    relation_results, rela_losses, rela_metrics = self.relation_heads(instance_detections,instance_local_features,
                                                                                      to_pred_phrases,to_pred_phrase_local_features,
                                                                                      to_pred_crowds,crowd_local_features,None,None,training=training,iteration=iteration)

                    losses.update(rela_losses)
                else:
                    losses = {}
            else:
                losses={}
            if not training:
                instance_results = self._postprocess(instance_detections, batched_inputs, images.image_sizes,"instance")
                for i, r in enumerate(instance_results):
                    if i >= len(results):
                        results.append({})
                    if relation_results:
                        results[i]['phrases']=relation_results[i]
                    results[i].update(instance_results[i])

        return results, losses, metrics

    def generate_phrase_instance_dict(self, pred_instances, pred_phrases):
        phrase_instance_dict_list=[]
        max_len_list=[]
        for image_i in range(len(pred_phrases)):
            pred_instance=pred_instances[image_i]
            pred_phrase=pred_phrases[image_i]

            phrase_instance_io = compute_boxes_io(pred_phrase.pred_boxes.tensor, pred_instance.pred_boxes.tensor)
            phrase_ids, instance_ids = torch.where(phrase_instance_io > 0.5)
            instance_classes = pred_instance.pred_classes[instance_ids]

            phrase_instance = defaultdict(dict)
            max_len = 0
            for phrase_id, instance_id, instance_class in zip(phrase_ids, instance_ids, instance_classes):
                phrase_id_item = str(phrase_id.item())
                instance_class_item = str(instance_class.item())
                inner_key = phrase_id_item+"_"+instance_class_item
                if inner_key not in phrase_instance[phrase_id_item]:
                    phrase_instance[phrase_id_item][inner_key] = []
                phrase_instance[phrase_id_item][inner_key].append(instance_id)
                max_len = max(max_len, len(phrase_instance[phrase_id_item][inner_key]))

            phrase_instance_dict_list.append(phrase_instance)
            max_len_list.append(max_len)
        return phrase_instance_dict_list,max_len_list

@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results