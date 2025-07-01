# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import time
from collections import defaultdict
import logging
import math
import numpy as np
import copy
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)
from matplotlib import pyplot as plt

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
# import pydensecrf.densecrf as dcrf
from fvcore.nn import smooth_l1_loss

import detectron2.utils.comm as comm
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, Triplets
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.utils.torch_utils import box_iou, GAT, GCN

from .instance_encoder import build_instance_encoder
from .instance_head import build_instance_head
from .predicate_head import build_predicate_head
from .pair_head import build_pair_head
from .triplet_head import build_triplet_head
from detr.transformer import CrossTransformerEncoderLayer2, CrossTransformerEncoderLayer3, GroupTransformer, GroupTransformer2, PhraseTransformer, PhraseTransformerEncoder, PhraseTransformerEncoderLayer
import copy
RELATION_HEADS_REGISTRY = Registry("RELATION_HEADS")
RELATION_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_relation_heads(cfg):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.RELATION_HEADS.NAME
    return RELATION_HEADS_REGISTRY.get(name)(cfg)

class RelationHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg):
        super(RelationHeads, self).__init__()

        self.device = cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.relation_num = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST

        feature_types=["language"] # 300->512
        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)[:80, :]
        self.semantic_embed = nn.Embedding(self.instance_num, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        self.language_fc = nn.Linear(300, 512)

        instance_feature_types=["instance_visual", # 1024->256*2
                                "instance_location",] # 4->256*2
        self.visual_fc = nn.Linear(1024, 512)
        self.location_fc = nn.Linear(4, 512)
        self.visual_lstm = nn.LSTM(512, 256, num_layers=1, batch_first=True, bidirectional=True)  # 1024->512
        self.location_lstm = nn.LSTM(512, 256, num_layers=1, batch_first=True, bidirectional=True)  # 1024->512


        crowd_feature_types=[]
        if "crowd" in self.relation_head_list:
            crowd_feature_types += ["crowd_visual", # 1024->512
                                    "crowd_location",] # 4->512

        phrase_feature_types = []
        if "phrase" in self.relation_head_list:
            phrase_feature_types += ["phrase_visual"] # 1024->512


        self.in_crowd_use_crowd=cfg.MODEL.RELATION_HEADS.IN_CROWD_USE_CROWD and ("crowd" in self.relation_head_list)

        self.use_attention_feature = cfg.MODEL.RELATION_HEADS.USE_ATTENTION_FEATURE
        self.attention = cfg.MODEL.RELATION_HEADS.ATTENTION
        self.minus = cfg.MODEL.RELATION_HEADS.MINUS
        if self.minus:
            self.predicate_fc1 = nn.Linear(512 * (len(feature_types)+len(instance_feature_types)+len(crowd_feature_types)+len(phrase_feature_types)), 512)
        else:
            self.predicate_fc1 = nn.Linear(512 * (2*len(feature_types)+2*len(instance_feature_types)+2*len(crowd_feature_types)+len(phrase_feature_types)), 512)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(512, self.relation_num+1)
        self.predicate_ac2 = nn.Sigmoid()

    def forward_language(self,instance_classes):
        instance_class_embeddings = self.semantic_embed(instance_classes)
        instance_language_out = F.relu(self.language_fc(instance_class_embeddings))
        return instance_language_out

    def forward_visual(self,raw_visual_features):
        visual_features = self.visual_fc(raw_visual_features)
        return visual_features

    def forward_location(self,instance_boxes,phrase_boxes):
        if len(instance_boxes.shape)==3:
            phrase_union_widths = phrase_boxes[:, :, 2] - phrase_boxes[:, :, 0]
            phrase_union_heights = phrase_boxes[:, :, 3] - phrase_boxes[:, :, 1]
            instance_locations = torch.stack([(instance_boxes[:, :, 0] - phrase_boxes[:, :, 0]) / phrase_union_widths,
                                              (instance_boxes[:, :, 1] - phrase_boxes[:, :, 1]) / phrase_union_heights,
                                              (instance_boxes[:, :, 2] - phrase_boxes[:, :, 2]) / phrase_union_widths,
                                              (instance_boxes[:, :, 3] - phrase_boxes[:, :, 3]) / phrase_union_heights], dim=2)
        elif len(instance_boxes.shape)==2:
            phrase_union_widths = phrase_boxes[:, 2] - phrase_boxes[:, 0]
            phrase_union_heights = phrase_boxes[:, 3] - phrase_boxes[:, 1]
            instance_locations = torch.stack([(instance_boxes[:, 0] - phrase_boxes[:, 0]) / phrase_union_widths,
                                              (instance_boxes[:, 1] - phrase_boxes[:, 1]) / phrase_union_heights,
                                              (instance_boxes[:, 2] - phrase_boxes[:, 2]) / phrase_union_widths,
                                              (instance_boxes[:, 3] - phrase_boxes[:, 3]) / phrase_union_heights],
                                             dim=1)
        else:
            print(instance_boxes.shape)
            exit()
        location_features=self.location_fc(instance_locations)
        return location_features

    def forward_crowd(self,crowd_raw_visual_features,crowd_boxes,phrase_boxes):
        crowd_visual_features = self.forward_visual(crowd_raw_visual_features)
        crowd_location_features = self.forward_location(crowd_boxes,phrase_boxes)
        return crowd_visual_features,crowd_location_features

    def forward_phrase(self,phrase_raw_visual_features):
        phrase_visual_features = self.forward_visual(phrase_raw_visual_features)
        return phrase_visual_features

    def forward(self, pred_instances, pred_instance_features, to_pred_phrases, to_pred_phrase_features,
                pred_crowds=None, pred_crowd_features=None, pred_phrases=None, pred_phrase_features=None, training=True, iteration=1):
        raise NotImplemented()

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

@RELATION_HEADS_REGISTRY.register()
class StandardRelationHeads(RelationHeads):

    def __init__(self, cfg):
        super(StandardRelationHeads, self).__init__(cfg)
        self.in_crowd_lstm = nn.LSTM(512*2, 1, num_layers=1, batch_first=True)  # 1024->1
        self.in_crowd_ac = nn.Sigmoid()

    def forward_instances_with_in_crowd(self,instance_lens,instance_raw_visual_features,instance_boxes,phrase_boxes,use_crowd=False,crowd_visual_features=None,crowd_location_features=None):
        ## subject pack info
        instance_lens = instance_lens.cpu()
        sorted_instance_lens, instance_indices = torch.sort(instance_lens, descending=True)
        instance_indices_reverse = torch.sort(instance_indices)[1]

        ## subject instance visual features into lstm
        instance_visual_features=instance_visual_in_crowd_features=self.forward_visual(instance_raw_visual_features)
        if use_crowd:
            instance_visual_in_crowd_features = instance_visual_in_crowd_features - crowd_visual_features.unsqueeze(1).repeat(1,instance_visual_in_crowd_features.shape[1],1)
        packed_sorted_instance_visual_features = pack_padded_sequence(instance_visual_in_crowd_features[instance_indices],sorted_instance_lens, batch_first=True)
        packed_sorted_instance_visual_lstm_out = self.visual_lstm(packed_sorted_instance_visual_features)[0]
        sorted_instance_visual_lstm_out = pad_packed_sequence(packed_sorted_instance_visual_lstm_out, batch_first=True)[0]
        instance_visual_lstm_features = sorted_instance_visual_lstm_out[instance_indices_reverse]

        ## subject instance location features into lstm
        instance_location_features=instance_location_in_crowd_features = self.forward_location(instance_boxes,phrase_boxes.unsqueeze(1).repeat(1,instance_boxes.shape[1],1))
        if use_crowd:
            instance_location_in_crowd_features = instance_location_in_crowd_features - crowd_location_features.unsqueeze(1).repeat(1,instance_visual_in_crowd_features.shape[1],1)
        packed_sorted_instance_location_features = pack_padded_sequence(instance_location_in_crowd_features[instance_indices],sorted_instance_lens, batch_first=True)
        packed_sorted_instance_location_lstm_out = self.location_lstm(packed_sorted_instance_location_features)[0]
        sorted_instance_location_lstm_out = pad_packed_sequence(packed_sorted_instance_location_lstm_out, batch_first=True)[0]
        instance_location_lstm_features = sorted_instance_location_lstm_out[instance_indices_reverse]

        ## subject in crowd
        sorted_instance_lstm_out=torch.cat([sorted_instance_visual_lstm_out,sorted_instance_location_lstm_out],dim=2)
        sorted_instance_in_crowd_scores=self.in_crowd_ac(self.in_crowd_lstm(sorted_instance_lstm_out)[0])
        instance_in_crowd_scores=sorted_instance_in_crowd_scores[instance_indices_reverse]

        if self.use_attention_feature:
            return instance_visual_lstm_features, instance_location_lstm_features, instance_in_crowd_scores
        else:
            return instance_visual_features, instance_location_features, instance_in_crowd_scores


    def forward(self, pred_instances, pred_instance_features, to_pred_phrases, to_pred_phrase_features,
                pred_crowds=None, pred_crowd_features=None, pred_phrases=None, pred_phrase_features=None, training=True,iteration=1):
        losses = {}
        metrics = {}
        results = []
        for i in range(len(pred_instances)):
            pred_instance = pred_instances[i]
            pred_instance_boxes = pred_instance.pred_boxes.tensor
            pred_instance_feature = pred_instance_features[i]
            to_pred_phrase = to_pred_phrases[i]
            to_pred_phrase_boxes = to_pred_phrase.pred_boxes.tensor
            if "phrase" in self.relation_head_list:
                to_pred_phrase_feature=to_pred_phrase_features[i]
            if pred_crowd_features and len(pred_crowd_features)>0:
                pred_crowd = pred_crowds[i]
                pred_crowd_boxes = pred_crowd.pred_boxes.tensor
                pred_crowd_feature = pred_crowd_features[i]
            if pred_phrase_features and len(pred_phrase_features)>0:
                pred_phrase = pred_phrases[i]
                pred_phrase_boxes = pred_phrase.pred_boxes.tensor
                pred_phrase_feature = pred_phrase_features[i]

            ## language feature for relation prediction
            subject_language_features = self.forward_language(to_pred_phrase.pred_subject_classes)
            object_language_features = self.forward_language(to_pred_phrase.pred_object_classes)
            # print(subject_language_features[:, :10])
            # print(object_language_features[:, :10])
            if "phrase" in self.relation_head_list:
                phrase_visual_features = self.forward_phrase(to_pred_phrase_feature)
                # print("phrase")
                # print(phrase_visual_features[:,:10])
            if "crowd" in self.relation_head_list:
                subject_crowd_visual_features, subject_crowd_location_features = self.forward_crowd(
                    pred_crowd_feature[to_pred_phrase.pred_subject_crowd_ids],
                    pred_crowd_boxes[to_pred_phrase.pred_subject_crowd_ids],
                    to_pred_phrase_boxes)
                # print("crowd")
                # print(subject_crowd_visual_features[:,:10])
                # print(subject_crowd_location_features[:, :10])
                object_crowd_visual_features, object_crowd_location_features = self.forward_crowd(
                    pred_crowd_feature[to_pred_phrase.pred_object_crowd_ids],
                    pred_crowd_boxes[to_pred_phrase.pred_object_crowd_ids],
                    to_pred_phrase_boxes)
                # print(object_crowd_visual_features[:, :10])
                # print(object_crowd_location_features[:, :10])
                subject_visual_features, subject_location_features, \
                subject_in_crowd_scores=self.forward_instances_with_in_crowd(
                    to_pred_phrase.pred_subject_lens,
                    pred_instance_feature[to_pred_phrase.pred_subject_ids_list],
                    pred_instance_boxes[to_pred_phrase.pred_subject_ids_list],
                    to_pred_phrase_boxes,use_crowd=self.in_crowd_use_crowd,crowd_visual_features=subject_crowd_visual_features, crowd_location_features=subject_crowd_location_features
                    )
                object_visual_features, object_location_features, \
                object_in_crowd_scores=self.forward_instances_with_in_crowd(
                    to_pred_phrase.pred_object_lens,
                    pred_instance_feature[to_pred_phrase.pred_object_ids_list],
                    pred_instance_boxes[to_pred_phrase.pred_object_ids_list],
                    to_pred_phrase_boxes,use_crowd=self.in_crowd_use_crowd,crowd_visual_features=object_crowd_visual_features, crowd_location_features=object_crowd_location_features
                    )
                # print("instance")
                # print(subject_visual_features[:, :,:10])
                # print(subject_location_features[:, :, :10])
                # print(object_visual_features[:, :,:10])
                # print(object_location_features[:, :, :10])
            else:
                subject_visual_features, subject_location_features, \
                subject_in_crowd_scores=self.forward_instances_with_in_crowd(
                    to_pred_phrase.pred_subject_lens,
                    pred_instance_feature[to_pred_phrase.pred_subject_ids_list],
                    pred_instance_boxes[to_pred_phrase.pred_subject_ids_list],
                    to_pred_phrase_boxes,use_crowd=self.in_crowd_use_crowd
                    )
                object_visual_features, object_location_features, \
                object_in_crowd_scores=self.forward_instances_with_in_crowd(
                    to_pred_phrase.pred_object_lens,
                    pred_instance_feature[to_pred_phrase.pred_object_ids_list],
                    pred_instance_boxes[to_pred_phrase.pred_object_ids_list],
                    to_pred_phrase_boxes,use_crowd=self.in_crowd_use_crowd
                    )

            subject_instance_visual_features = []
            subject_instance_location_features = []
            subject_in_crowd_score_cat = []
            subject_in_crowd_gt_cat = []
            object_instance_visual_features = []
            object_instance_location_features = []
            object_in_crowd_score_cat = []
            object_in_crowd_gt_cat = []
            for i,phrase_i in enumerate(range(len(to_pred_phrase))):
                # print(to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]])
                subject_in_crowd_scores_phrase_i=subject_in_crowd_scores[phrase_i]*to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]]
                valid_subject_in_crowd_score = subject_in_crowd_scores_phrase_i[:to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_visual_feature = subject_visual_features[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_location_feature = subject_location_features[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]]

                object_in_crowd_scores_phrase_i = object_in_crowd_scores[phrase_i] * to_pred_phrase.pred_object_in_clust[phrase_i].unsqueeze(1)[:object_in_crowd_scores[phrase_i].shape[0]]
                valid_object_in_crowd_score = object_in_crowd_scores_phrase_i[:to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_visual_feature = object_visual_features[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_location_feature = object_location_features[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]]

                if self.attention:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature * valid_subject_in_crowd_score, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature * valid_subject_in_crowd_score, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature * valid_object_in_crowd_score, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature * valid_object_in_crowd_score, dim=0))
                else:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature, dim=0))

                subject_in_crowd_score_cat.append(valid_subject_in_crowd_score[:, 0])
                object_in_crowd_score_cat.append(valid_object_in_crowd_score[:, 0])
                if training:
                    subject_in_crowd_gt_cat.append(to_pred_phrase.pred_subject_gt_in_crowds_list[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]])
                    object_in_crowd_gt_cat.append(to_pred_phrase.pred_object_gt_in_crowds_list[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]])

            if training:
                for phrase_i in range(len(to_pred_phrase)):
                    if to_pred_phrase.confidence[phrase_i]==0:
                        subject_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        object_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        subject_in_crowd_gt_cat[phrase_i]=torch.Tensor([]).to(self.device)
                        object_in_crowd_gt_cat[phrase_i]=torch.Tensor([]).to(self.device)

            subject_crowd_out_cat = torch.cat(subject_in_crowd_score_cat)  # triplet num, instance num, 1
            object_crowd_out_cat = torch.cat(object_in_crowd_score_cat)
            if training:
                subject_crowd_gt_cat = torch.cat(subject_in_crowd_gt_cat)  # triplet num, instance num, 1
                object_crowd_gt_cat = torch.cat(object_in_crowd_gt_cat)  # triplet num, instance num, 1

            subject_instance_visual_features = torch.stack(subject_instance_visual_features)
            subject_instance_location_features = torch.stack(subject_instance_location_features)

            object_instance_visual_features = torch.stack(object_instance_visual_features)
            object_instance_location_features = torch.stack(object_instance_location_features)

            if self.minus:
                predicate_feature=torch.cat([subject_language_features-object_language_features,
                                             subject_instance_visual_features-object_instance_visual_features,
                                             subject_instance_location_features-object_instance_location_features],dim=1)
            else:
                predicate_feature=torch.cat([subject_language_features,object_language_features,
                                             subject_instance_visual_features,object_instance_visual_features,
                                             subject_instance_location_features,object_instance_location_features],dim=1)

            if "crowd" in self.relation_head_list:
                if self.minus:
                    predicate_feature = torch.cat([predicate_feature,
                                                   subject_crowd_visual_features-object_crowd_visual_features,
                                                   subject_crowd_location_features-object_crowd_location_features],dim=1)
                else:
                    predicate_feature=torch.cat([predicate_feature,
                                             subject_crowd_visual_features,object_crowd_visual_features,
                                             subject_crowd_location_features, object_crowd_location_features],dim=1)
            if "phrase" in self.relation_head_list:
                predicate_feature = torch.cat([predicate_feature,
                                               phrase_visual_features],dim=1)

            predicate_feature=self.predicate_ac1(self.predicate_fc1(predicate_feature))
            predicate_out = self.predicate_ac2(self.predicate_fc2(predicate_feature))

            if training:
                losses['loss_in_crowd_sub'] = F.binary_cross_entropy(
                                            subject_crowd_out_cat,
                                            subject_crowd_gt_cat.float(),
                                            # reduction="sum",
                                        )
                losses['loss_in_crowd_obj'] = F.binary_cross_entropy(
                                            object_crowd_out_cat,
                                            object_crowd_gt_cat.float(),
                                            # reduction="sum",
                                        )
                losses['loss_rel_pos'],losses['loss_rel_neg']=self.binary_focal_loss(predicate_out.flatten(), to_pred_phrase.gt_relation_onehots.flatten())
            else:
                to_pred_phrase.pred_subject_ids_list=to_pred_phrase.pred_subject_ids_list[:,:subject_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_object_ids_list = to_pred_phrase.pred_object_ids_list[:,:object_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_subject_in_crowds_list = subject_in_crowd_scores.squeeze(2)
                to_pred_phrase.pred_object_in_crowds_list = object_in_crowd_scores.squeeze(2)
                to_pred_phrase.pred_relation_scores=predicate_out
                results.append(to_pred_phrase)

        return results, losses, metrics

@RELATION_HEADS_REGISTRY.register()
class DifferentRelationHeads_CTrans2(nn.Module):

    def __init__(self, cfg):
        super(DifferentRelationHeads_CTrans2, self).__init__()

        self.device = cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.relation_num = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST

        feature_types = ["language"]  # 300->512
        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)[:80, :]
        self.semantic_embed = nn.Embedding(self.instance_num, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        self.language_fc = nn.Linear(300, cfg.MODEL.RELATION_HEADS.FEATURE_DIM)

        if cfg.MODEL.ROI_HEADS.ENCODE_DIM > 0:
            self.visual_fc = nn.Linear(cfg.MODEL.ROI_HEADS.ENCODE_DIM_OUT, cfg.MODEL.RELATION_HEADS.FEATURE_DIM)
        else:
            self.visual_fc = nn.Linear(cfg.MODEL.ROI_BOX_HEAD.FC_DIM, cfg.MODEL.RELATION_HEADS.FEATURE_DIM)
        self.location_fc = nn.Linear(4, cfg.MODEL.RELATION_HEADS.FEATURE_DIM)

        self.transformerEncoderLayer=CrossTransformerEncoderLayer2(d_model=cfg.MODEL.RELATION_HEADS.FEATURE_DIM*2,nhead=8)


        instance_feature_types=["instance_visual", # 1024->256*2
                                "instance_location",] # 4->256*2

        crowd_feature_types=[]
        if "crowd" in self.relation_head_list:
            crowd_feature_types += ["crowd_visual", # 1024->512
                                    "crowd_location",] # 4->512

        phrase_feature_types = []
        if "phrase" in self.relation_head_list:
            phrase_feature_types += ["phrase_visual"] # 1024->512


        self.in_crowd_use_crowd=cfg.MODEL.RELATION_HEADS.IN_CROWD_USE_CROWD and ("crowd" in self.relation_head_list)
        self.use_attention_feature = cfg.MODEL.RELATION_HEADS.USE_ATTENTION_FEATURE
        self.attention = cfg.MODEL.RELATION_HEADS.ATTENTION
        self.minus = cfg.MODEL.RELATION_HEADS.MINUS
        self.crowd_loss= cfg.MODEL.RELATION_HEADS.CROWD_LOSS
        self.relative=cfg.MODEL.RELATION_HEADS.RELATIVE
        self.reweight=cfg.MODEL.RELATION_HEADS.REWEIGHT
        self.reweight_after=cfg.MODEL.RELATION_HEADS.REWEIGHT_AFTER
        fre_statistics=np.load("statistics.npy",allow_pickle=True).item()
        self.fre_statistics=torch.FloatTensor([1]+fre_statistics['relation_frequency_list']).to(self.device)

        self.in_crowd_fc1 = nn.Linear(cfg.MODEL.RELATION_HEADS.FEATURE_DIM*2,256)
        self.in_crowd_fc2 = nn.Linear(256, 1)

        if self.minus:
            self.predicate_fc1 = nn.Linear(cfg.MODEL.RELATION_HEADS.FEATURE_DIM * (len(feature_types)+len(instance_feature_types)+len(crowd_feature_types)+len(phrase_feature_types)), cfg.MODEL.RELATION_HEADS.FEATURE_DIM)
        else:
            self.predicate_fc1 = nn.Linear(cfg.MODEL.RELATION_HEADS.FEATURE_DIM * (2*len(feature_types)+2*len(instance_feature_types)+2*len(crowd_feature_types)+len(phrase_feature_types)), cfg.MODEL.RELATION_HEADS.FEATURE_DIM)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(cfg.MODEL.RELATION_HEADS.FEATURE_DIM, self.relation_num+1)
        self.predicate_ac2 = nn.Sigmoid()

        if cfg.MODEL.RELATION_HEADS.FEATURE_INIT=="normal_constant":
            nn.init.normal_(self.language_fc.weight, mean=0, std=0.01)
            nn.init.constant_(self.language_fc.bias, 0)
            nn.init.normal_(self.visual_fc.weight, mean=0, std=0.01)
            nn.init.constant_(self.visual_fc.bias, 0)
            nn.init.normal_(self.location_fc.weight, mean=0, std=0.01)
            nn.init.constant_(self.location_fc.bias, 0)

            nn.init.normal_(self.in_crowd_fc1.weight, mean=0, std=0.01)
            nn.init.constant_(self.in_crowd_fc1.bias, 0)
            nn.init.normal_(self.in_crowd_fc2.weight, mean=0, std=0.01)
            nn.init.constant_(self.in_crowd_fc2.bias, 0)
            nn.init.normal_(self.predicate_fc1.weight, mean=0, std=0.01)
            nn.init.constant_(self.predicate_fc1.bias, 0)

            nn.init.normal_(self.predicate_fc2.weight, mean=0, std=0.01)
            nn.init.constant_(self.predicate_fc2.bias, 0)
        else:
            exit("NO SUCH INIT")
            exit()

    def forward_language(self,instance_classes):
        instance_class_embeddings = self.semantic_embed(instance_classes)
        instance_language_out = F.relu(self.language_fc(instance_class_embeddings))
        return instance_language_out

    def forward_visual(self,raw_visual_features):
        visual_features = self.visual_fc(raw_visual_features)
        return visual_features

    def forward_location(self,instance_boxes,phrase_boxes):
        if len(instance_boxes.shape)==3:
            phrase_union_widths = phrase_boxes[:, :, 2] - phrase_boxes[:, :, 0]
            phrase_union_heights = phrase_boxes[:, :, 3] - phrase_boxes[:, :, 1]
            instance_locations = torch.stack([(instance_boxes[:, :, 0] - phrase_boxes[:, :, 0]) / phrase_union_widths,
                                              (instance_boxes[:, :, 1] - phrase_boxes[:, :, 1]) / phrase_union_heights,
                                              (instance_boxes[:, :, 2] - phrase_boxes[:, :, 2]) / phrase_union_widths,
                                              (instance_boxes[:, :, 3] - phrase_boxes[:, :, 3]) / phrase_union_heights], dim=2)
        elif len(instance_boxes.shape)==2:
            phrase_union_widths = phrase_boxes[:, 2] - phrase_boxes[:, 0]
            phrase_union_heights = phrase_boxes[:, 3] - phrase_boxes[:, 1]
            instance_locations = torch.stack([(instance_boxes[:, 0] - phrase_boxes[:, 0]) / phrase_union_widths,
                                              (instance_boxes[:, 1] - phrase_boxes[:, 1]) / phrase_union_heights,
                                              (instance_boxes[:, 2] - phrase_boxes[:, 2]) / phrase_union_widths,
                                              (instance_boxes[:, 3] - phrase_boxes[:, 3]) / phrase_union_heights],
                                             dim=1)
        else:
            print(instance_boxes.shape)
            exit()
        location_features=self.location_fc(instance_locations)
        return location_features

    def forward_instance(self, instance_raw_visual_features,instance_boxes,phrase_boxes):
        instance_visual_features=self.forward_visual(instance_raw_visual_features)
        instance_location_features = self.forward_location(instance_boxes,phrase_boxes.unsqueeze(1).repeat(1,instance_boxes.shape[1],1))
        return instance_visual_features,instance_location_features

    def forward_crowd(self,crowd_raw_visual_features,crowd_boxes,phrase_boxes):
        crowd_visual_features = self.forward_visual(crowd_raw_visual_features)
        crowd_location_features = self.forward_location(crowd_boxes,phrase_boxes)
        return crowd_visual_features,crowd_location_features

    def forward_phrase(self,phrase_raw_visual_features):
        phrase_visual_features = self.forward_visual(phrase_raw_visual_features)
        return phrase_visual_features

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

    def forward(self, pred_instances, pred_instance_features,
                to_pred_crowds, to_pred_crowd_features, to_pred_crowd_fusion_features,
                to_pred_phrases, to_pred_phrase_features,
                training=True,iteration=1):
        losses = {}
        metrics = {}
        results = []

        subject_crowd_out_cat_batch=[]
        object_crowd_out_cat_batch=[]
        subject_crowd_gt_cat_batch=[]
        object_crowd_gt_cat_batch = []
        predicate_out_batch=[]
        gt_relation_onehots_batch=[]
        for i in range(len(pred_instances)):
            pred_instance = pred_instances[i]
            pred_instance_boxes = pred_instance.pred_boxes.tensor
            pred_instance_feature = pred_instance_features[i]
            to_pred_phrase = to_pred_phrases[i]
            to_pred_phrase_boxes = to_pred_phrase.pred_boxes.tensor
            if "phrase" in self.relation_head_list:
                to_pred_phrase_feature=to_pred_phrase_features[i]
            if to_pred_crowd_features and len(to_pred_crowd_features)>0:
                to_pred_crowd = to_pred_crowds[i]
                to_pred_crowd_boxes = to_pred_crowd.pred_boxes.tensor
                to_pred_crowd_feature = to_pred_crowd_features[i]
            if to_pred_phrase_features and len(to_pred_phrase_features)>0:
                to_pred_phrase = to_pred_phrases[i]
                to_pred_phrase_boxes = to_pred_phrase.pred_boxes.tensor
                to_pred_phrase_feature = to_pred_phrase_features[i]

            ## language feature for relation prediction
            subject_language_features = self.forward_language(to_pred_phrase.pred_subject_classes)
            object_language_features = self.forward_language(to_pred_phrase.pred_object_classes)

            ## phrase
            if "phrase" in self.relation_head_list:
                phrase_visual_features = self.forward_phrase(to_pred_phrase_feature)

            ## crowd
            if "crowd" in self.relation_head_list:
                subject_crowd_visual_features, subject_crowd_location_features = self.forward_crowd(
                    to_pred_crowd_feature[to_pred_phrase.pred_subject_crowd_ids],
                    to_pred_crowd_boxes[to_pred_phrase.pred_subject_crowd_ids],
                    to_pred_phrase_boxes)
                object_crowd_visual_features, object_crowd_location_features = self.forward_crowd(
                    to_pred_crowd_feature[to_pred_phrase.pred_object_crowd_ids],
                    to_pred_crowd_boxes[to_pred_phrase.pred_object_crowd_ids],
                    to_pred_phrase_boxes)

            subject_visual_features,subject_location_features=self.forward_instance(
                pred_instance_feature[to_pred_phrase.pred_subject_ids_list],
                pred_instance_boxes[to_pred_phrase.pred_subject_ids_list],
                to_pred_phrase_boxes
            )
            object_visual_features,object_location_features=self.forward_instance(
                pred_instance_feature[to_pred_phrase.pred_object_ids_list],
                pred_instance_boxes[to_pred_phrase.pred_object_ids_list],
                to_pred_phrase_boxes
            )

            if training:
                encode_subject_features = self.transformerEncoderLayer(
                    real_src = torch.cat([subject_visual_features,subject_location_features],dim=-1).transpose(1,0), # max_len, phrase, feature_dim
                    gt_src = torch.cat([object_visual_features,object_location_features],dim=-1).transpose(1,0),
                    gt_src_mask = None,
                    gt_src_key_padding_mask = to_pred_phrase.pred_object_gt_in_crowds_list==0,
                    printsize = False, need_sigmoid_weights = False).transpose(1,0)
                # print(to_pred_phrase.pred_subject_ids_list)
                # print(to_pred_phrase.pred_subject_gt_in_crowds_list)
                # print(encode_subject_features.shape)
                # print("---------------")
                encode_object_features = self.transformerEncoderLayer(
                    real_src = torch.cat([object_visual_features,object_location_features],dim=-1).transpose(1,0),
                    gt_src = torch.cat([subject_visual_features,subject_location_features],dim=-1).transpose(1,0),
                    gt_src_mask = None,
                    gt_src_key_padding_mask = to_pred_phrase.pred_subject_gt_in_crowds_list==0,
                    printsize = False, need_sigmoid_weights = False).transpose(1,0)
                # print(to_pred_phrase.pred_object_ids_list)
                # print(to_pred_phrase.pred_object_gt_in_crowds_list)
                # print(encode_object_features.shape)
                # print("===============")
                # exit()
            else:
                encode_subject_features = self.transformerEncoderLayer(
                    real_src = torch.cat([subject_visual_features,subject_location_features],dim=-1).transpose(1,0),
                    gt_src = torch.cat([object_visual_features,object_location_features],dim=-1).transpose(1,0),
                    gt_src_mask = None,
                    gt_src_key_padding_mask = to_pred_phrase.pred_object_ids_list<0,
                    printsize = False, need_sigmoid_weights = False).transpose(1,0)
                encode_object_features = self.transformerEncoderLayer(
                    real_src = torch.cat([object_visual_features,object_location_features],dim=-1).transpose(1,0),
                    gt_src = torch.cat([subject_visual_features,subject_location_features],dim=-1).transpose(1,0),
                    gt_src_mask = None,
                    gt_src_key_padding_mask = to_pred_phrase.pred_subject_ids_list<0,
                    printsize = False, need_sigmoid_weights = False).transpose(1,0)

            subject_in_crowd_scores = F.sigmoid(self.in_crowd_fc2(F.relu(self.in_crowd_fc1(encode_subject_features))))
            object_in_crowd_scores = F.sigmoid(self.in_crowd_fc2(F.relu(self.in_crowd_fc1(encode_object_features))))

            subject_instance_visual_features = []
            subject_instance_location_features = []
            subject_in_crowd_score_cat = []
            subject_in_crowd_gt_cat = []
            object_instance_visual_features = []
            object_instance_location_features = []
            object_in_crowd_score_cat = []
            object_in_crowd_gt_cat = []
            for i,phrase_i in enumerate(range(len(to_pred_phrase))):
                # print(to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]])
                subject_in_crowd_scores_phrase_i=subject_in_crowd_scores[phrase_i]*to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]]
                valid_subject_in_crowd_score = subject_in_crowd_scores_phrase_i[:to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_visual_feature = subject_visual_features[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_location_feature = subject_location_features[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]]

                object_in_crowd_scores_phrase_i = object_in_crowd_scores[phrase_i] * to_pred_phrase.pred_object_in_clust[phrase_i].unsqueeze(1)[:object_in_crowd_scores[phrase_i].shape[0]]
                valid_object_in_crowd_score = object_in_crowd_scores_phrase_i[:to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_visual_feature = object_visual_features[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_location_feature = object_location_features[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]]

                if self.attention:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature * valid_subject_in_crowd_score, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature * valid_subject_in_crowd_score, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature * valid_object_in_crowd_score, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature * valid_object_in_crowd_score, dim=0))
                else:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature, dim=0))

                subject_in_crowd_score_cat.append(valid_subject_in_crowd_score[:, 0])
                object_in_crowd_score_cat.append(valid_object_in_crowd_score[:, 0])
                if training:
                    subject_in_crowd_gt_cat.append(to_pred_phrase.pred_subject_gt_in_crowds_list[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]])
                    object_in_crowd_gt_cat.append(to_pred_phrase.pred_object_gt_in_crowds_list[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]])

            if training:
                for phrase_i in range(len(to_pred_phrase)):
                    if to_pred_phrase.confidence[phrase_i]==0:
                        subject_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        object_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        subject_in_crowd_gt_cat[phrase_i]=torch.Tensor([]).to(self.device)
                        object_in_crowd_gt_cat[phrase_i]=torch.Tensor([]).to(self.device)

            subject_crowd_out_cat = torch.cat(subject_in_crowd_score_cat)  # triplet num, instance num, 1
            object_crowd_out_cat = torch.cat(object_in_crowd_score_cat)
            if training:
                subject_crowd_gt_cat = torch.cat(subject_in_crowd_gt_cat)  # triplet num, instance num, 1
                object_crowd_gt_cat = torch.cat(object_in_crowd_gt_cat)  # triplet num, instance num, 1

            subject_instance_visual_features = torch.stack(subject_instance_visual_features)
            subject_instance_location_features = torch.stack(subject_instance_location_features)

            object_instance_visual_features = torch.stack(object_instance_visual_features)
            object_instance_location_features = torch.stack(object_instance_location_features)

            if self.minus:
                predicate_feature=torch.cat([subject_language_features-object_language_features,
                                             subject_instance_visual_features-object_instance_visual_features,
                                             subject_instance_location_features-object_instance_location_features],dim=1)
            else:
                predicate_feature=torch.cat([subject_language_features,object_language_features,
                                             subject_instance_visual_features,object_instance_visual_features,
                                             subject_instance_location_features,object_instance_location_features],dim=1)

            if "crowd" in self.relation_head_list:
                if self.minus:
                    predicate_feature = torch.cat([predicate_feature,
                                                   subject_crowd_visual_features-object_crowd_visual_features,
                                                   subject_crowd_location_features-object_crowd_location_features],dim=1)
                else:
                    predicate_feature=torch.cat([predicate_feature,
                                             subject_crowd_visual_features,object_crowd_visual_features,
                                             subject_crowd_location_features, object_crowd_location_features],dim=1)
            if "phrase" in self.relation_head_list:
                predicate_feature = torch.cat([predicate_feature, phrase_visual_features],dim=1)

            predicate_feature=self.predicate_ac1(self.predicate_fc1(predicate_feature))
            predicate_dist = self.predicate_fc2(predicate_feature)
            if self.reweight>0:
                if self.reweight_after:
                    predicate_out = self.predicate_ac2(predicate_dist) - self.reweight * self.fre_statistics.unsqueeze(0).repeat(predicate_dist.shape[0], 1)
                else:
                    predicate_out = self.predicate_ac2(predicate_dist - self.reweight * self.fre_statistics.unsqueeze(0).repeat(predicate_dist.shape[0], 1))
            else:
                predicate_out = self.predicate_ac2(predicate_dist)
            if training:
                subject_crowd_out_cat_batch.append(subject_crowd_out_cat)
                object_crowd_out_cat_batch.append(object_crowd_out_cat)
                subject_crowd_gt_cat_batch.append(subject_crowd_gt_cat.float())
                object_crowd_gt_cat_batch.append(object_crowd_gt_cat.float())
                # print("subject_crowd_out_cat",subject_crowd_out_cat)
                # print("subject_crowd_gt_cat",subject_crowd_gt_cat)
                predicate_out_batch.append(predicate_out.flatten())
                gt_relation_onehots_batch.append(to_pred_phrase.gt_relation_onehots.flatten())
                # print("predicate_out",predicate_out)
                # print("gt_relation_onehots",to_pred_phrase.gt_relation_onehots)
                # print("------------------------------")
            else:
                subject_in_crowd_scores = subject_in_crowd_scores.squeeze(2) * to_pred_phrase.pred_subject_in_clust
                object_in_crowd_scores = object_in_crowd_scores.squeeze(2) * to_pred_phrase.pred_object_in_clust
                if self.relative:
                    subject_in_crowd_scores = subject_in_crowd_scores / (torch.max(subject_in_crowd_scores, dim=1)[0].unsqueeze(-1) + 1e-10)
                    object_in_crowd_scores = object_in_crowd_scores / (torch.max(object_in_crowd_scores, dim=1)[0].unsqueeze(-1) + 1e-10)

                to_pred_phrase.pred_subject_ids_list=to_pred_phrase.pred_subject_ids_list[:,:subject_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_object_ids_list = to_pred_phrase.pred_object_ids_list[:,:object_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_subject_in_crowds_list = subject_in_crowd_scores
                to_pred_phrase.pred_object_in_crowds_list = object_in_crowd_scores
                to_pred_phrase.pred_relation_scores=predicate_out
                results.append(to_pred_phrase)

        if training:
            subject_crowd_out_cat_batch = torch.cat(subject_crowd_out_cat_batch, dim=-1)
            object_crowd_out_cat_batch = torch.cat(object_crowd_out_cat_batch, dim=-1)
            subject_crowd_gt_cat_batch = torch.cat(subject_crowd_gt_cat_batch, dim=-1)
            object_crowd_gt_cat_batch = torch.cat(object_crowd_gt_cat_batch, dim=-1)
            predicate_out_batch = torch.cat(predicate_out_batch, dim=-1)
            gt_relation_onehots_batch = torch.cat(gt_relation_onehots_batch, dim=-1)

            if self.crowd_loss=="bce":
                losses['loss_in_crowd_sub'] = F.binary_cross_entropy_with_logits(
                                            subject_crowd_out_cat_batch,
                                            subject_crowd_gt_cat_batch,
                                            # reduction="sum",
                                        )
                losses['loss_in_crowd_obj'] = F.binary_cross_entropy_with_logits(
                                            object_crowd_out_cat_batch,
                                            object_crowd_gt_cat_batch,
                                            # reduction="sum",
                                        )
            elif self.crowd_loss=="bifocal":
                losses['loss_in_crowd_sub_pos'],losses['loss_in_crowd_sub_neg']=self.binary_focal_loss(subject_crowd_out_cat_batch, subject_crowd_gt_cat_batch,pos_gamma=2.0,neg_gamma=1.0)
                losses['loss_in_crowd_obj_pos'], losses['loss_in_crowd_obj_neg'] = self.binary_focal_loss(object_crowd_out_cat_batch, object_crowd_gt_cat_batch, pos_gamma=2.0, neg_gamma=1.0)
            else:
                print("NO SUCH CROWD LOSS",self.crowd_loss)
                exit()
            losses['loss_rel_pos'],losses['loss_rel_neg']=self.binary_focal_loss(predicate_out_batch, gt_relation_onehots_batch)

        return results, losses, metrics

@RELATION_HEADS_REGISTRY.register()
class DifferentRelationHeads_CTrans2NoGP(nn.Module):

    def __init__(self, cfg):
        super(DifferentRelationHeads_CTrans2NoGP, self).__init__()

        self.device = cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.relation_num = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST

        feature_types = ["language"]  # 300->512
        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)[:80, :]
        self.semantic_embed = nn.Embedding(self.instance_num, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        self.language_fc = nn.Linear(300, 512)
        self.visual_fc = nn.Linear(1024, 512)
        self.location_fc = nn.Linear(4, 512)

        nn.init.normal_(self.language_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.language_fc.bias, 0)
        nn.init.normal_(self.visual_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.visual_fc.bias, 0)
        nn.init.normal_(self.location_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.location_fc.bias, 0)

        instance_feature_types = ["instance_visual",  # 1024->256*2
                                  "instance_location", ]  # 4->256*2

        crowd_feature_types = []
        if "crowd" in self.relation_head_list:
            crowd_feature_types += ["crowd_visual",  # 1024->512
                                    "crowd_location", ]  # 4->512

        phrase_feature_types = []
        if "phrase" in self.relation_head_list:
            phrase_feature_types += ["phrase_visual"]  # 1024->512

        self.in_crowd_use_crowd = cfg.MODEL.RELATION_HEADS.IN_CROWD_USE_CROWD and (
                    "crowd" in self.relation_head_list)
        self.use_attention_feature = cfg.MODEL.RELATION_HEADS.USE_ATTENTION_FEATURE
        self.attention = cfg.MODEL.RELATION_HEADS.ATTENTION
        self.minus = cfg.MODEL.RELATION_HEADS.MINUS
        self.reweight = cfg.MODEL.RELATION_HEADS.REWEIGHT
        self.reweight_after = cfg.MODEL.RELATION_HEADS.REWEIGHT_AFTER
        fre_statistics = np.load("statistics.npy", allow_pickle=True).item()
        self.fre_statistics = torch.FloatTensor([1] + fre_statistics['relation_frequency_list']).to(self.device)

        if self.minus:
            self.predicate_fc1 = nn.Linear(512 * (
                        len(feature_types) + len(instance_feature_types) + len(crowd_feature_types) + len(
                    phrase_feature_types)), 512)
        else:
            self.predicate_fc1 = nn.Linear(512 * (
                        2 * len(feature_types) + 2 * len(instance_feature_types) + 2 * len(
                    crowd_feature_types) + len(phrase_feature_types)), 512)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(512, self.relation_num + 1)
        self.predicate_ac2 = nn.Sigmoid()
        nn.init.normal_(self.predicate_fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.predicate_fc1.bias, 0)
        nn.init.normal_(self.predicate_fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.predicate_fc2.bias, 0)

    def forward_language(self, instance_classes):
        instance_class_embeddings = self.semantic_embed(instance_classes)
        instance_language_out = F.relu(self.language_fc(instance_class_embeddings))
        return instance_language_out

    def forward_visual(self, raw_visual_features):
        visual_features = self.visual_fc(raw_visual_features)
        return visual_features

    def forward_location(self, instance_boxes, phrase_boxes):
        if len(instance_boxes.shape) == 3:
            phrase_union_widths = phrase_boxes[:, :, 2] - phrase_boxes[:, :, 0]
            phrase_union_heights = phrase_boxes[:, :, 3] - phrase_boxes[:, :, 1]
            instance_locations = torch.stack(
                [(instance_boxes[:, :, 0] - phrase_boxes[:, :, 0]) / phrase_union_widths,
                 (instance_boxes[:, :, 1] - phrase_boxes[:, :, 1]) / phrase_union_heights,
                 (instance_boxes[:, :, 2] - phrase_boxes[:, :, 2]) / phrase_union_widths,
                 (instance_boxes[:, :, 3] - phrase_boxes[:, :, 3]) / phrase_union_heights], dim=2)
        elif len(instance_boxes.shape) == 2:
            phrase_union_widths = phrase_boxes[:, 2] - phrase_boxes[:, 0]
            phrase_union_heights = phrase_boxes[:, 3] - phrase_boxes[:, 1]
            instance_locations = torch.stack([(instance_boxes[:, 0] - phrase_boxes[:, 0]) / phrase_union_widths,
                                              (instance_boxes[:, 1] - phrase_boxes[:, 1]) / phrase_union_heights,
                                              (instance_boxes[:, 2] - phrase_boxes[:, 2]) / phrase_union_widths,
                                              (instance_boxes[:, 3] - phrase_boxes[:, 3]) / phrase_union_heights],
                                             dim=1)
        else:
            print(instance_boxes.shape)
            exit()
        location_features = self.location_fc(instance_locations)
        return location_features

    def forward_instance(self, instance_raw_visual_features, instance_boxes, phrase_boxes):
        instance_visual_features = self.forward_visual(instance_raw_visual_features)
        instance_location_features = self.forward_location(instance_boxes, phrase_boxes.unsqueeze(1).repeat(1,
                                                                                                            instance_boxes.shape[
                                                                                                                1],
                                                                                                            1))
        return instance_visual_features, instance_location_features

    def forward_crowd(self, crowd_raw_visual_features, crowd_boxes, phrase_boxes):
        crowd_visual_features = self.forward_visual(crowd_raw_visual_features)
        crowd_location_features = self.forward_location(crowd_boxes, phrase_boxes)
        return crowd_visual_features, crowd_location_features

    def forward_phrase(self, phrase_raw_visual_features):
        phrase_visual_features = self.forward_visual(phrase_raw_visual_features)
        return phrase_visual_features

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

    def forward(self, pred_instances, pred_instance_features,
                to_pred_crowds, to_pred_crowd_features, to_pred_crowd_fusion_features,
                to_pred_phrases, to_pred_phrase_features,
                training=True,iteration=1):
        losses = {}
        metrics = {}
        results = []
        for i in range(len(pred_instances)):
            pred_instance = pred_instances[i]
            pred_instance_boxes = pred_instance.pred_boxes.tensor
            pred_instance_feature = pred_instance_features[i]
            to_pred_phrase = to_pred_phrases[i]
            to_pred_phrase_boxes = to_pred_phrase.pred_boxes.tensor
            if "phrase" in self.relation_head_list:
                to_pred_phrase_feature = to_pred_phrase_features[i]
            if to_pred_crowd_features and len(to_pred_crowd_features) > 0:
                pred_crowd = to_pred_crowds[i]
                pred_crowd_boxes = pred_crowd.pred_boxes.tensor
                pred_crowd_feature = to_pred_crowd_features[i]

            ## language feature for relation prediction
            subject_language_features = self.forward_language(to_pred_phrase.pred_subject_classes)
            object_language_features = self.forward_language(to_pred_phrase.pred_object_classes)

            ## phrase
            if "phrase" in self.relation_head_list:
                phrase_visual_features = self.forward_phrase(to_pred_phrase_feature)

            ## crowd
            if "crowd" in self.relation_head_list:
                subject_crowd_visual_features, subject_crowd_location_features = self.forward_crowd(
                    pred_crowd_feature[to_pred_phrase.pred_subject_crowd_ids],
                    pred_crowd_boxes[to_pred_phrase.pred_subject_crowd_ids],
                    to_pred_phrase_boxes)
                object_crowd_visual_features, object_crowd_location_features = self.forward_crowd(
                    pred_crowd_feature[to_pred_phrase.pred_object_crowd_ids],
                    pred_crowd_boxes[to_pred_phrase.pred_object_crowd_ids],
                    to_pred_phrase_boxes)

            subject_visual_features, subject_location_features = self.forward_instance(
                pred_instance_feature[to_pred_phrase.pred_subject_ids_list],
                pred_instance_boxes[to_pred_phrase.pred_subject_ids_list],
                to_pred_phrase_boxes
            )
            object_visual_features, object_location_features = self.forward_instance(
                pred_instance_feature[to_pred_phrase.pred_object_ids_list],
                pred_instance_boxes[to_pred_phrase.pred_object_ids_list],
                to_pred_phrase_boxes
            )

            subject_in_crowd_scores = torch.ones_like(to_pred_phrase.pred_subject_ids_list).unsqueeze(2)
            object_in_crowd_scores = torch.ones_like(to_pred_phrase.pred_object_ids_list).unsqueeze(2)

            subject_instance_visual_features = []
            subject_instance_location_features = []
            subject_in_crowd_score_cat = []
            subject_in_crowd_gt_cat = []
            object_instance_visual_features = []
            object_instance_location_features = []
            object_in_crowd_score_cat = []
            object_in_crowd_gt_cat = []
            for i, phrase_i in enumerate(range(len(to_pred_phrase))):
                # print(to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]])
                subject_in_crowd_scores_phrase_i = subject_in_crowd_scores[phrase_i] * \
                                                   to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[
                                                   :subject_in_crowd_scores[phrase_i].shape[0]]
                valid_subject_in_crowd_score = subject_in_crowd_scores_phrase_i[
                                               :to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_visual_feature = subject_visual_features[phrase_i][
                                               :to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_location_feature = subject_location_features[phrase_i][
                                                 :to_pred_phrase.pred_subject_lens[phrase_i]]

                object_in_crowd_scores_phrase_i = object_in_crowd_scores[phrase_i] * \
                                                  to_pred_phrase.pred_object_in_clust[phrase_i].unsqueeze(1)[
                                                  :object_in_crowd_scores[phrase_i].shape[0]]
                valid_object_in_crowd_score = object_in_crowd_scores_phrase_i[
                                              :to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_visual_feature = object_visual_features[phrase_i][
                                              :to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_location_feature = object_location_features[phrase_i][
                                                :to_pred_phrase.pred_object_lens[phrase_i]]

                if self.attention:
                    subject_instance_visual_features.append(
                        torch.mean(valid_subject_visual_feature * valid_subject_in_crowd_score, dim=0))
                    subject_instance_location_features.append(
                        torch.mean(valid_subject_location_feature * valid_subject_in_crowd_score, dim=0))
                    object_instance_visual_features.append(
                        torch.mean(valid_object_visual_feature * valid_object_in_crowd_score, dim=0))
                    object_instance_location_features.append(
                        torch.mean(valid_object_location_feature * valid_object_in_crowd_score, dim=0))
                else:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature, dim=0))

                subject_in_crowd_score_cat.append(valid_subject_in_crowd_score[:, 0])
                object_in_crowd_score_cat.append(valid_object_in_crowd_score[:, 0])
                if training:
                    subject_in_crowd_gt_cat.append(to_pred_phrase.pred_subject_gt_in_crowds_list[phrase_i][
                                                   :to_pred_phrase.pred_subject_lens[phrase_i]])
                    object_in_crowd_gt_cat.append(to_pred_phrase.pred_object_gt_in_crowds_list[phrase_i][
                                                  :to_pred_phrase.pred_object_lens[phrase_i]])

            if training:
                for phrase_i in range(len(to_pred_phrase)):
                    if to_pred_phrase.confidence[phrase_i] == 0:
                        subject_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        object_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        subject_in_crowd_gt_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        object_in_crowd_gt_cat[phrase_i] = torch.Tensor([]).to(self.device)

            subject_crowd_out_cat = torch.cat(subject_in_crowd_score_cat)  # triplet num, instance num, 1
            object_crowd_out_cat = torch.cat(object_in_crowd_score_cat)
            if training:
                subject_crowd_gt_cat = torch.cat(subject_in_crowd_gt_cat)  # triplet num, instance num, 1
                object_crowd_gt_cat = torch.cat(object_in_crowd_gt_cat)  # triplet num, instance num, 1

            subject_instance_visual_features = torch.stack(subject_instance_visual_features)
            subject_instance_location_features = torch.stack(subject_instance_location_features)

            object_instance_visual_features = torch.stack(object_instance_visual_features)
            object_instance_location_features = torch.stack(object_instance_location_features)

            if self.minus:
                predicate_feature = torch.cat([subject_language_features - object_language_features,
                                               subject_instance_visual_features - object_instance_visual_features,
                                               subject_instance_location_features - object_instance_location_features],
                                              dim=1)
            else:
                predicate_feature = torch.cat([subject_language_features, object_language_features,
                                               subject_instance_visual_features, object_instance_visual_features,
                                               subject_instance_location_features,
                                               object_instance_location_features], dim=1)

            if "crowd" in self.relation_head_list:
                if self.minus:
                    predicate_feature = torch.cat([predicate_feature,
                                                   subject_crowd_visual_features - object_crowd_visual_features,
                                                   subject_crowd_location_features - object_crowd_location_features],
                                                  dim=1)
                else:
                    predicate_feature = torch.cat([predicate_feature,
                                                   subject_crowd_visual_features, object_crowd_visual_features,
                                                   subject_crowd_location_features, object_crowd_location_features],
                                                  dim=1)
            if "phrase" in self.relation_head_list:
                predicate_feature = torch.cat([predicate_feature,
                                               phrase_visual_features], dim=1)

            predicate_feature = self.predicate_ac1(self.predicate_fc1(predicate_feature))
            predicate_dist = self.predicate_fc2(predicate_feature)

            if self.reweight>0:
                if self.reweight_after:
                    predicate_out = self.predicate_ac2(predicate_dist) - self.reweight * self.fre_statistics.unsqueeze(0).repeat(predicate_dist.shape[0], 1)
                else:
                    predicate_out = self.predicate_ac2(predicate_dist - self.reweight * self.fre_statistics.unsqueeze(0).repeat(predicate_dist.shape[0], 1))
            else:
                predicate_out = self.predicate_ac2(predicate_dist)

            if training:
                losses['loss_in_crowd_sub'] = F.binary_cross_entropy(
                    subject_crowd_out_cat.float(),
                    subject_crowd_gt_cat.float(),
                    # reduction="sum",
                )
                losses['loss_in_crowd_obj'] = F.binary_cross_entropy(
                    object_crowd_out_cat.float(),
                    object_crowd_gt_cat.float(),
                    # reduction="sum",
                )
                losses['loss_rel_pos'], losses['loss_rel_neg'] = self.binary_focal_loss(predicate_out.flatten(),
                                                                                        to_pred_phrase.gt_relation_onehots.flatten())
            else:
                to_pred_phrase.pred_subject_ids_list = to_pred_phrase.pred_subject_ids_list[:,
                                                       :subject_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_object_ids_list = to_pred_phrase.pred_object_ids_list[:,
                                                      :object_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_subject_in_crowds_list = subject_in_crowd_scores.squeeze(2)
                to_pred_phrase.pred_object_in_crowds_list = object_in_crowd_scores.squeeze(2)
                to_pred_phrase.pred_relation_scores = predicate_out
                results.append(to_pred_phrase)

        return results, losses, metrics

@RELATION_HEADS_REGISTRY.register()
class DifferentRelationHeads_BiLSTM(nn.Module):

    def __init__(self, cfg):
        super(DifferentRelationHeads_BiLSTM, self).__init__()

        self.device = cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.relation_num = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST

        feature_types = ["language"]  # 300->512
        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)[:80, :]
        self.semantic_embed = nn.Embedding(self.instance_num, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        self.language_fc = nn.Linear(300, 512)
        self.visual_fc = nn.Linear(1024, 512)
        self.location_fc = nn.Linear(4, 512)

        nn.init.normal_(self.language_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.language_fc.bias, 0)
        nn.init.normal_(self.visual_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.visual_fc.bias, 0)
        nn.init.normal_(self.location_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.location_fc.bias, 0)

        self.visual_lstm = nn.LSTM(512, 256, num_layers=1, batch_first=True, bidirectional=True)  # 1024->512
        self.location_lstm = nn.LSTM(512, 256, num_layers=1, batch_first=True, bidirectional=True)  # 1024->512

        instance_feature_types=["instance_visual", # 1024->256*2
                                "instance_location",] # 4->256*2

        crowd_feature_types=[]
        if "crowd" in self.relation_head_list:
            crowd_feature_types += ["crowd_visual", # 1024->512
                                    "crowd_location",] # 4->512

        phrase_feature_types = []
        if "phrase" in self.relation_head_list:
            phrase_feature_types += ["phrase_visual"] # 1024->512


        self.in_crowd_use_crowd=cfg.MODEL.RELATION_HEADS.IN_CROWD_USE_CROWD and ("crowd" in self.relation_head_list)
        self.use_attention_feature = cfg.MODEL.RELATION_HEADS.USE_ATTENTION_FEATURE
        self.attention = cfg.MODEL.RELATION_HEADS.ATTENTION
        self.minus = cfg.MODEL.RELATION_HEADS.MINUS
        self.reweight = cfg.MODEL.RELATION_HEADS.REWEIGHT
        self.reweight_after = cfg.MODEL.RELATION_HEADS.REWEIGHT_AFTER
        fre_statistics = np.load("statistics.npy", allow_pickle=True).item()
        self.fre_statistics = torch.FloatTensor([1] + fre_statistics['relation_frequency_list']).to(self.device)

        self.in_crowd_fc1 = nn.Linear(512*2,512)
        self.in_crowd_fc2 = nn.Linear(512, 1)
        nn.init.normal_(self.in_crowd_fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.in_crowd_fc1.bias, 0)
        nn.init.normal_(self.in_crowd_fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.in_crowd_fc2.bias, 0)

        if self.minus:
            self.predicate_fc1 = nn.Linear(512 * (len(feature_types)+len(instance_feature_types)+len(crowd_feature_types)+len(phrase_feature_types)), 512)
        else:
            self.predicate_fc1 = nn.Linear(512 * (2*len(feature_types)+2*len(instance_feature_types)+2*len(crowd_feature_types)+len(phrase_feature_types)), 512)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(512, self.relation_num+1)
        self.predicate_ac2 = nn.Sigmoid()
        nn.init.normal_(self.predicate_fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.predicate_fc1.bias, 0)
        nn.init.normal_(self.predicate_fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.predicate_fc2.bias, 0)

    def forward_language(self,instance_classes):
        instance_class_embeddings = self.semantic_embed(instance_classes)
        instance_language_out = F.relu(self.language_fc(instance_class_embeddings))
        return instance_language_out

    def forward_visual(self,raw_visual_features):
        visual_features = self.visual_fc(raw_visual_features)
        return visual_features

    def forward_location(self,instance_boxes,phrase_boxes):
        if len(instance_boxes.shape)==3:
            phrase_union_widths = phrase_boxes[:, :, 2] - phrase_boxes[:, :, 0]
            phrase_union_heights = phrase_boxes[:, :, 3] - phrase_boxes[:, :, 1]
            instance_locations = torch.stack([(instance_boxes[:, :, 0] - phrase_boxes[:, :, 0]) / phrase_union_widths,
                                              (instance_boxes[:, :, 1] - phrase_boxes[:, :, 1]) / phrase_union_heights,
                                              (instance_boxes[:, :, 2] - phrase_boxes[:, :, 2]) / phrase_union_widths,
                                              (instance_boxes[:, :, 3] - phrase_boxes[:, :, 3]) / phrase_union_heights], dim=2)
        elif len(instance_boxes.shape)==2:
            phrase_union_widths = phrase_boxes[:, 2] - phrase_boxes[:, 0]
            phrase_union_heights = phrase_boxes[:, 3] - phrase_boxes[:, 1]
            instance_locations = torch.stack([(instance_boxes[:, 0] - phrase_boxes[:, 0]) / phrase_union_widths,
                                              (instance_boxes[:, 1] - phrase_boxes[:, 1]) / phrase_union_heights,
                                              (instance_boxes[:, 2] - phrase_boxes[:, 2]) / phrase_union_widths,
                                              (instance_boxes[:, 3] - phrase_boxes[:, 3]) / phrase_union_heights],
                                             dim=1)
        else:
            print(instance_boxes.shape)
            exit()
        location_features=self.location_fc(instance_locations)
        return location_features

    def forward_crowd(self,crowd_raw_visual_features,crowd_boxes,phrase_boxes):
        crowd_visual_features = self.forward_visual(crowd_raw_visual_features)
        crowd_location_features = self.forward_location(crowd_boxes,phrase_boxes)
        return crowd_visual_features,crowd_location_features

    def forward_phrase(self,phrase_raw_visual_features):
        phrase_visual_features = self.forward_visual(phrase_raw_visual_features)
        return phrase_visual_features

    def forward_instances_with_in_crowd(self,instance_lens,instance_raw_visual_features,instance_boxes,phrase_boxes,use_crowd=False,crowd_visual_features=None,crowd_location_features=None):
        ## subject pack info
        instance_lens = instance_lens.cpu()
        sorted_instance_lens, instance_indices = torch.sort(instance_lens, descending=True)
        instance_indices_reverse = torch.sort(instance_indices)[1]

        ## subject instance visual features into lstm
        instance_visual_features=instance_visual_in_crowd_features=self.forward_visual(instance_raw_visual_features)
        if use_crowd:
            instance_visual_in_crowd_features = instance_visual_in_crowd_features - crowd_visual_features.unsqueeze(1).repeat(1,instance_visual_in_crowd_features.shape[1],1)
        packed_sorted_instance_visual_features = pack_padded_sequence(instance_visual_in_crowd_features[instance_indices],sorted_instance_lens, batch_first=True)
        packed_sorted_instance_visual_lstm_out = self.visual_lstm(packed_sorted_instance_visual_features)[0]
        sorted_instance_visual_lstm_out = pad_packed_sequence(packed_sorted_instance_visual_lstm_out, batch_first=True)[0]
        instance_visual_lstm_features = sorted_instance_visual_lstm_out[instance_indices_reverse]

        ## subject instance location features into lstm
        instance_location_features=instance_location_in_crowd_features = self.forward_location(instance_boxes,phrase_boxes.unsqueeze(1).repeat(1,instance_boxes.shape[1],1))
        if use_crowd:
            instance_location_in_crowd_features = instance_location_in_crowd_features - crowd_location_features.unsqueeze(1).repeat(1,instance_visual_in_crowd_features.shape[1],1)
        packed_sorted_instance_location_features = pack_padded_sequence(instance_location_in_crowd_features[instance_indices],sorted_instance_lens, batch_first=True)
        packed_sorted_instance_location_lstm_out = self.location_lstm(packed_sorted_instance_location_features)[0]
        sorted_instance_location_lstm_out = pad_packed_sequence(packed_sorted_instance_location_lstm_out, batch_first=True)[0]
        instance_location_lstm_features = sorted_instance_location_lstm_out[instance_indices_reverse]

        return instance_visual_features, instance_location_features,instance_visual_lstm_features, instance_location_lstm_features

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

    def forward(self, pred_instances, pred_instance_features,
                to_pred_crowds, to_pred_crowd_features, to_pred_crowd_fusion_features,
                to_pred_phrases, to_pred_phrase_features,
                training=True,iteration=1):
        losses = {}
        metrics = {}
        results = []
        for i in range(len(pred_instances)):
            pred_instance = pred_instances[i]
            pred_instance_boxes = pred_instance.pred_boxes.tensor
            pred_instance_feature = pred_instance_features[i]
            to_pred_phrase = to_pred_phrases[i]
            to_pred_phrase_boxes = to_pred_phrase.pred_boxes.tensor
            if "phrase" in self.relation_head_list:
                to_pred_phrase_feature=to_pred_phrase_features[i]
            if to_pred_crowd_features and len(to_pred_crowd_features)>0:
                pred_crowd = to_pred_crowds[i]
                pred_crowd_boxes = pred_crowd.pred_boxes.tensor
                pred_crowd_feature = to_pred_crowd_features[i]

            ## language feature for relation prediction
            subject_language_features = self.forward_language(to_pred_phrase.pred_subject_classes)
            object_language_features = self.forward_language(to_pred_phrase.pred_object_classes)

            ## phrase
            if "phrase" in self.relation_head_list:
                phrase_visual_features = self.forward_phrase(to_pred_phrase_feature)

            ## crowd
            if "crowd" in self.relation_head_list:
                subject_crowd_visual_features, subject_crowd_location_features = self.forward_crowd(
                    pred_crowd_feature[to_pred_phrase.pred_subject_crowd_ids],
                    pred_crowd_boxes[to_pred_phrase.pred_subject_crowd_ids],
                    to_pred_phrase_boxes)
                object_crowd_visual_features, object_crowd_location_features = self.forward_crowd(
                    pred_crowd_feature[to_pred_phrase.pred_object_crowd_ids],
                    pred_crowd_boxes[to_pred_phrase.pred_object_crowd_ids],
                    to_pred_phrase_boxes)

            ## instance
            subject_visual_features, subject_location_features,subject_visual_lstm_features, subject_location_lstm_features=self.forward_instances_with_in_crowd(
                to_pred_phrase.pred_subject_lens,
                pred_instance_feature[to_pred_phrase.pred_subject_ids_list],
                pred_instance_boxes[to_pred_phrase.pred_subject_ids_list],
                to_pred_phrase_boxes,use_crowd=self.in_crowd_use_crowd
                )
            object_visual_features, object_location_features,object_visual_lstm_features, object_location_lstm_features=self.forward_instances_with_in_crowd(
                to_pred_phrase.pred_object_lens,
                pred_instance_feature[to_pred_phrase.pred_object_ids_list],
                pred_instance_boxes[to_pred_phrase.pred_object_ids_list],
                to_pred_phrase_boxes,use_crowd=self.in_crowd_use_crowd
                )

            ## in crowd
            # subject_lstm_out = torch.cat([subject_visual_lstm_features-object_crowd_visual_features.unsqueeze(1).repeat(1,subject_visual_lstm_features.shape[1],1), subject_location_lstm_features-object_crowd_location_features.unsqueeze(1).repeat(1,subject_visual_lstm_features.shape[1],1)],dim=2)
            # object_lstm_out = torch.cat([object_visual_lstm_features-subject_crowd_visual_features.unsqueeze(1).repeat(1,object_visual_lstm_features.shape[1],1), object_location_lstm_features-subject_crowd_visual_features.unsqueeze(1).repeat(1,object_visual_lstm_features.shape[1],1)], dim=2)
            subject_lstm_out = torch.cat([subject_visual_lstm_features,subject_location_lstm_features], dim=2)
            object_lstm_out = torch.cat([object_visual_lstm_features,object_location_lstm_features], dim=2)
            subject_in_crowd_scores = F.sigmoid(self.in_crowd_fc2(F.relu(self.in_crowd_fc1(subject_lstm_out))))
            object_in_crowd_scores = F.sigmoid(self.in_crowd_fc2(F.relu(self.in_crowd_fc1(object_lstm_out))))

            subject_instance_visual_features = []
            subject_instance_location_features = []
            subject_in_crowd_score_cat = []
            subject_in_crowd_gt_cat = []
            object_instance_visual_features = []
            object_instance_location_features = []
            object_in_crowd_score_cat = []
            object_in_crowd_gt_cat = []
            for i,phrase_i in enumerate(range(len(to_pred_phrase))):
                # print(to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]])
                subject_in_crowd_scores_phrase_i=subject_in_crowd_scores[phrase_i]*to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]]
                valid_subject_in_crowd_score = subject_in_crowd_scores_phrase_i[:to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_visual_feature = subject_visual_features[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_location_feature = subject_location_features[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]]

                object_in_crowd_scores_phrase_i = object_in_crowd_scores[phrase_i] * to_pred_phrase.pred_object_in_clust[phrase_i].unsqueeze(1)[:object_in_crowd_scores[phrase_i].shape[0]]
                valid_object_in_crowd_score = object_in_crowd_scores_phrase_i[:to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_visual_feature = object_visual_features[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_location_feature = object_location_features[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]]

                if self.attention:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature * valid_subject_in_crowd_score, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature * valid_subject_in_crowd_score, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature * valid_object_in_crowd_score, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature * valid_object_in_crowd_score, dim=0))
                else:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature, dim=0))

                subject_in_crowd_score_cat.append(valid_subject_in_crowd_score[:, 0])
                object_in_crowd_score_cat.append(valid_object_in_crowd_score[:, 0])
                if training:
                    subject_in_crowd_gt_cat.append(to_pred_phrase.pred_subject_gt_in_crowds_list[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]])
                    object_in_crowd_gt_cat.append(to_pred_phrase.pred_object_gt_in_crowds_list[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]])

            if training:
                for phrase_i in range(len(to_pred_phrase)):
                    if to_pred_phrase.confidence[phrase_i]==0:
                        subject_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        object_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        subject_in_crowd_gt_cat[phrase_i]=torch.Tensor([]).to(self.device)
                        object_in_crowd_gt_cat[phrase_i]=torch.Tensor([]).to(self.device)
                #     else:
                #         print(subject_in_crowd_gt_cat[phrase_i])
                #         print(subject_in_crowd_score_cat[phrase_i])
                # print()
            # for phrase_i in range(len(to_pred_phrase)):
            #     print(subject_in_crowd_score_cat[phrase_i])
            # print()
            subject_crowd_out_cat = torch.cat(subject_in_crowd_score_cat)  # triplet num, instance num, 1
            object_crowd_out_cat = torch.cat(object_in_crowd_score_cat)
            if training:
                subject_crowd_gt_cat = torch.cat(subject_in_crowd_gt_cat)  # triplet num, instance num, 1
                object_crowd_gt_cat = torch.cat(object_in_crowd_gt_cat)  # triplet num, instance num, 1

            subject_instance_visual_features = torch.stack(subject_instance_visual_features)
            subject_instance_location_features = torch.stack(subject_instance_location_features)

            object_instance_visual_features = torch.stack(object_instance_visual_features)
            object_instance_location_features = torch.stack(object_instance_location_features)

            if self.minus:
                predicate_feature=torch.cat([subject_language_features-object_language_features,
                                             subject_instance_visual_features-object_instance_visual_features,
                                             subject_instance_location_features-object_instance_location_features],dim=1)
            else:
                predicate_feature=torch.cat([subject_language_features,object_language_features,
                                             subject_instance_visual_features,object_instance_visual_features,
                                             subject_instance_location_features,object_instance_location_features],dim=1)

            if "crowd" in self.relation_head_list:
                if self.minus:
                    predicate_feature = torch.cat([predicate_feature,
                                                   subject_crowd_visual_features-object_crowd_visual_features,
                                                   subject_crowd_location_features-object_crowd_location_features],dim=1)
                else:
                    predicate_feature=torch.cat([predicate_feature,
                                             subject_crowd_visual_features,object_crowd_visual_features,
                                             subject_crowd_location_features, object_crowd_location_features],dim=1)
            if "phrase" in self.relation_head_list:
                predicate_feature = torch.cat([predicate_feature,
                                               phrase_visual_features],dim=1)

            predicate_feature=self.predicate_ac1(self.predicate_fc1(predicate_feature))
            predicate_dist = self.predicate_fc2(predicate_feature)
            if self.reweight>0:
                if self.reweight_after:
                    predicate_out = self.predicate_ac2(predicate_dist) - self.reweight * self.fre_statistics.unsqueeze(0).repeat(predicate_dist.shape[0], 1)
                else:
                    predicate_out = self.predicate_ac2(predicate_dist - self.reweight * self.fre_statistics.unsqueeze(0).repeat(predicate_dist.shape[0], 1))
            else:
                predicate_out = self.predicate_ac2(predicate_dist)
            # print(predicate_out)
            if training:
                losses['loss_in_crowd_sub'] = F.binary_cross_entropy(
                                            subject_crowd_out_cat,
                                            subject_crowd_gt_cat.float(),
                                            # reduction="sum",
                                        )
                losses['loss_in_crowd_obj'] = F.binary_cross_entropy(
                                            object_crowd_out_cat,
                                            object_crowd_gt_cat.float(),
                                            # reduction="sum",
                                        )
                losses['loss_rel_pos'],losses['loss_rel_neg']=self.binary_focal_loss(predicate_out.flatten(), to_pred_phrase.gt_relation_onehots.flatten())
            else:
                to_pred_phrase.pred_subject_ids_list=to_pred_phrase.pred_subject_ids_list[:,:subject_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_object_ids_list = to_pred_phrase.pred_object_ids_list[:,:object_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_subject_in_crowds_list = subject_in_crowd_scores.squeeze(2)
                to_pred_phrase.pred_object_in_crowds_list = object_in_crowd_scores.squeeze(2)
                to_pred_phrase.pred_relation_scores=predicate_out
                results.append(to_pred_phrase)

        return results, losses, metrics

@RELATION_HEADS_REGISTRY.register()
class DifferentRelationHeads_GCN(nn.Module):

    def __init__(self, cfg):
        super(DifferentRelationHeads_GCN, self).__init__()

        self.device = cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.relation_num = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST

        feature_types = ["language"]  # 300->512
        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)[:80, :]
        self.semantic_embed = nn.Embedding(self.instance_num, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        self.language_fc = nn.Linear(300, 512)
        self.visual_fc = nn.Linear(1024, 512)
        self.location_fc = nn.Linear(4, 512)

        nn.init.normal_(self.language_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.language_fc.bias, 0)
        nn.init.normal_(self.visual_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.visual_fc.bias, 0)
        nn.init.normal_(self.location_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.location_fc.bias, 0)

        self.visual_encode = GCN(512, 512, nout=512)  # 1024->512
        self.location_encode = GCN(512, 512, nout=512)  # 1024->512

        instance_feature_types = ["instance_visual",  # 1024->256*2
                                  "instance_location", ]  # 4->256*2
        crowd_feature_types = []
        if "crowd" in self.relation_head_list:
            crowd_feature_types += ["crowd_visual",  # 1024->512
                                    "crowd_location", ]  # 4->512
        phrase_feature_types = []
        if "phrase" in self.relation_head_list:
            phrase_feature_types += ["phrase_visual"]  # 1024->512

        self.in_crowd_use_crowd = cfg.MODEL.RELATION_HEADS.IN_CROWD_USE_CROWD and ("crowd" in self.relation_head_list)
        self.use_attention_feature = cfg.MODEL.RELATION_HEADS.USE_ATTENTION_FEATURE
        self.attention = cfg.MODEL.RELATION_HEADS.ATTENTION
        self.minus = cfg.MODEL.RELATION_HEADS.MINUS
        self.reweight = cfg.MODEL.RELATION_HEADS.REWEIGHT
        self.reweight_after = cfg.MODEL.RELATION_HEADS.REWEIGHT_AFTER
        fre_statistics = np.load("statistics.npy", allow_pickle=True).item()
        self.fre_statistics = torch.FloatTensor([1] + fre_statistics['relation_frequency_list']).to(self.device)

        self.in_crowd_fc1 = nn.Linear(512 * 2, 512)
        self.in_crowd_fc2 = nn.Linear(512, 1)
        nn.init.normal_(self.in_crowd_fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.in_crowd_fc1.bias, 0)
        nn.init.normal_(self.in_crowd_fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.in_crowd_fc2.bias, 0)

        if self.minus:
            self.predicate_fc1 = nn.Linear(512 * (
                    len(feature_types) + len(instance_feature_types) + len(crowd_feature_types) + len(
                phrase_feature_types)), 512)
        else:
            self.predicate_fc1 = nn.Linear(512 * (
                    2 * len(feature_types) + 2 * len(instance_feature_types) + 2 * len(crowd_feature_types) + len(
                phrase_feature_types)), 512)
        self.predicate_fc2 = nn.Linear(512, self.relation_num + 1)
        nn.init.normal_(self.predicate_fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.predicate_fc1.bias, 0)
        nn.init.normal_(self.predicate_fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.predicate_fc2.bias, 0)

    def forward_language(self,instance_classes):
        instance_class_embeddings = self.semantic_embed(instance_classes)
        instance_language_out = F.relu(self.language_fc(instance_class_embeddings))
        return instance_language_out

    def forward_visual(self,raw_visual_features):
        visual_features = self.visual_fc(raw_visual_features)
        return visual_features

    def forward_location(self,instance_boxes,phrase_boxes):
        if len(instance_boxes.shape)==3:
            phrase_union_widths = phrase_boxes[:, :, 2] - phrase_boxes[:, :, 0]
            phrase_union_heights = phrase_boxes[:, :, 3] - phrase_boxes[:, :, 1]
            instance_locations = torch.stack([(instance_boxes[:, :, 0] - phrase_boxes[:, :, 0]) / phrase_union_widths,
                                              (instance_boxes[:, :, 1] - phrase_boxes[:, :, 1]) / phrase_union_heights,
                                              (instance_boxes[:, :, 2] - phrase_boxes[:, :, 2]) / phrase_union_widths,
                                              (instance_boxes[:, :, 3] - phrase_boxes[:, :, 3]) / phrase_union_heights], dim=2)
        elif len(instance_boxes.shape)==2:
            phrase_union_widths = phrase_boxes[:, 2] - phrase_boxes[:, 0]
            phrase_union_heights = phrase_boxes[:, 3] - phrase_boxes[:, 1]
            instance_locations = torch.stack([(instance_boxes[:, 0] - phrase_boxes[:, 0]) / phrase_union_widths,
                                              (instance_boxes[:, 1] - phrase_boxes[:, 1]) / phrase_union_heights,
                                              (instance_boxes[:, 2] - phrase_boxes[:, 2]) / phrase_union_widths,
                                              (instance_boxes[:, 3] - phrase_boxes[:, 3]) / phrase_union_heights],
                                             dim=1)
        else:
            print(instance_boxes.shape)
            exit()
        location_features=self.location_fc(instance_locations)
        return location_features

    def forward_crowd(self,crowd_raw_visual_features,crowd_boxes,phrase_boxes):
        crowd_visual_features = self.forward_visual(crowd_raw_visual_features)
        crowd_location_features = self.forward_location(crowd_boxes,phrase_boxes)
        return crowd_visual_features,crowd_location_features

    def forward_phrase(self,phrase_raw_visual_features):
        phrase_visual_features = self.forward_visual(phrase_raw_visual_features)
        return phrase_visual_features

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

    def forward_instances_with_in_crowd(self,instance_ids_list,instance_raw_visual_features,instance_boxes,phrase_boxes,use_crowd=False,crowd_visual_features=None,crowd_location_features=None):
        instance_ids_list_connect = (instance_ids_list >= 0).float()
        instance_ids_list_connect_matrix = torch.bmm(instance_ids_list_connect.unsqueeze(2),
                                                     instance_ids_list_connect.unsqueeze(1))

        ## subject instance visual features into gat
        instance_visual_features=instance_visual_in_crowd_features=self.forward_visual(instance_raw_visual_features)
        if use_crowd:
            instance_visual_in_crowd_features = instance_visual_in_crowd_features - crowd_visual_features.unsqueeze(1).repeat(1,instance_visual_in_crowd_features.shape[1],1)
        instance_visual_gat_features = self.visual_encode(instance_visual_in_crowd_features,instance_ids_list_connect_matrix)

        ## subject instance location features into lstm
        instance_location_features=instance_location_in_crowd_features = self.forward_location(instance_boxes,phrase_boxes.unsqueeze(1).repeat(1,instance_boxes.shape[1],1))
        if use_crowd:
            instance_location_in_crowd_features = instance_location_in_crowd_features - crowd_location_features.unsqueeze(1).repeat(1,instance_visual_in_crowd_features.shape[1],1)
        instance_location_gat_features = self.location_encode(instance_location_in_crowd_features,instance_ids_list_connect_matrix)

        return instance_visual_features, instance_location_features,instance_visual_gat_features, instance_location_gat_features

    def forward(self, pred_instances, pred_instance_features,
                to_pred_crowds, to_pred_crowd_features, to_pred_crowd_fusion_features,
                to_pred_phrases, to_pred_phrase_features,
                training=True,iteration=1):
        losses = {}
        metrics = {}
        results = []
        for i in range(len(pred_instances)):
            pred_instance = pred_instances[i]
            pred_instance_boxes = pred_instance.pred_boxes.tensor
            pred_instance_feature = pred_instance_features[i]
            to_pred_phrase = to_pred_phrases[i]
            to_pred_phrase_boxes = to_pred_phrase.pred_boxes.tensor
            if "phrase" in self.relation_head_list:
                to_pred_phrase_feature=to_pred_phrase_features[i]
            if to_pred_crowd_features and len(to_pred_crowd_features)>0:
                pred_crowd = to_pred_crowds[i]
                pred_crowd_boxes = pred_crowd.pred_boxes.tensor
                pred_crowd_feature = to_pred_crowd_features[i]


            ## language feature for relation prediction
            subject_language_features = self.forward_language(to_pred_phrase.pred_subject_classes)
            object_language_features = self.forward_language(to_pred_phrase.pred_object_classes)

            ## phrase
            if "phrase" in self.relation_head_list:
                phrase_visual_features = self.forward_phrase(to_pred_phrase_feature)

            ## crowd
            if "crowd" in self.relation_head_list:
                subject_crowd_visual_features, subject_crowd_location_features = self.forward_crowd(
                    pred_crowd_feature[to_pred_phrase.pred_subject_crowd_ids],
                    pred_crowd_boxes[to_pred_phrase.pred_subject_crowd_ids],
                    to_pred_phrase_boxes)
                object_crowd_visual_features, object_crowd_location_features = self.forward_crowd(
                    pred_crowd_feature[to_pred_phrase.pred_object_crowd_ids],
                    pred_crowd_boxes[to_pred_phrase.pred_object_crowd_ids],
                    to_pred_phrase_boxes)

            ## instance
            subject_visual_features, subject_location_features,subject_visual_lstm_features, subject_location_lstm_features=self.forward_instances_with_in_crowd(
                to_pred_phrase.pred_subject_ids_list,
                pred_instance_feature[to_pred_phrase.pred_subject_ids_list],
                pred_instance_boxes[to_pred_phrase.pred_subject_ids_list],
                to_pred_phrase_boxes,use_crowd=self.in_crowd_use_crowd
                )
            object_visual_features, object_location_features,object_visual_lstm_features, object_location_lstm_features=self.forward_instances_with_in_crowd(
                to_pred_phrase.pred_object_ids_list,
                pred_instance_feature[to_pred_phrase.pred_object_ids_list],
                pred_instance_boxes[to_pred_phrase.pred_object_ids_list],
                to_pred_phrase_boxes,use_crowd=self.in_crowd_use_crowd
                )

            ## in crowd
            # subject_lstm_out = torch.cat([subject_visual_lstm_features-object_crowd_visual_features.unsqueeze(1).repeat(1,subject_visual_lstm_features.shape[1],1), subject_location_lstm_features-object_crowd_location_features.unsqueeze(1).repeat(1,subject_visual_lstm_features.shape[1],1)],dim=2)
            # object_lstm_out = torch.cat([object_visual_lstm_features-subject_crowd_visual_features.unsqueeze(1).repeat(1,object_visual_lstm_features.shape[1],1), object_location_lstm_features-subject_crowd_visual_features.unsqueeze(1).repeat(1,object_visual_lstm_features.shape[1],1)], dim=2)
            subject_lstm_out = torch.cat([subject_visual_lstm_features,subject_location_lstm_features], dim=2)
            object_lstm_out = torch.cat([object_visual_lstm_features,object_location_lstm_features], dim=2)
            subject_in_crowd_scores = F.sigmoid(self.in_crowd_fc2(F.relu(self.in_crowd_fc1(subject_lstm_out))))
            object_in_crowd_scores = F.sigmoid(self.in_crowd_fc2(F.relu(self.in_crowd_fc1(object_lstm_out))))

            subject_instance_visual_features = []
            subject_instance_location_features = []
            subject_in_crowd_score_cat = []
            subject_in_crowd_gt_cat = []
            object_instance_visual_features = []
            object_instance_location_features = []
            object_in_crowd_score_cat = []
            object_in_crowd_gt_cat = []
            for i,phrase_i in enumerate(range(len(to_pred_phrase))):
                # print(to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]])
                subject_in_crowd_scores_phrase_i=subject_in_crowd_scores[phrase_i]*to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]]
                valid_subject_in_crowd_score = subject_in_crowd_scores_phrase_i[:to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_visual_feature = subject_visual_features[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_location_feature = subject_location_features[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]]

                object_in_crowd_scores_phrase_i = object_in_crowd_scores[phrase_i] * to_pred_phrase.pred_object_in_clust[phrase_i].unsqueeze(1)[:object_in_crowd_scores[phrase_i].shape[0]]
                valid_object_in_crowd_score = object_in_crowd_scores_phrase_i[:to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_visual_feature = object_visual_features[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_location_feature = object_location_features[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]]

                if self.attention:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature * valid_subject_in_crowd_score, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature * valid_subject_in_crowd_score, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature * valid_object_in_crowd_score, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature * valid_object_in_crowd_score, dim=0))
                else:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature, dim=0))

                subject_in_crowd_score_cat.append(valid_subject_in_crowd_score[:, 0])
                object_in_crowd_score_cat.append(valid_object_in_crowd_score[:, 0])
                if training:
                    subject_in_crowd_gt_cat.append(to_pred_phrase.pred_subject_gt_in_crowds_list[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]])
                    object_in_crowd_gt_cat.append(to_pred_phrase.pred_object_gt_in_crowds_list[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]])

            if training:
                for phrase_i in range(len(to_pred_phrase)):
                    if to_pred_phrase.confidence[phrase_i]==0:
                        subject_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        object_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        subject_in_crowd_gt_cat[phrase_i]=torch.Tensor([]).to(self.device)
                        object_in_crowd_gt_cat[phrase_i]=torch.Tensor([]).to(self.device)
                #     else:
                #         print(subject_in_crowd_gt_cat[phrase_i])
                #         print(subject_in_crowd_score_cat[phrase_i])
                # print()
            # for phrase_i in range(len(to_pred_phrase)):
            #     print(subject_in_crowd_score_cat[phrase_i])
            # print()
            subject_crowd_out_cat = torch.cat(subject_in_crowd_score_cat)  # triplet num, instance num, 1
            object_crowd_out_cat = torch.cat(object_in_crowd_score_cat)
            if training:
                subject_crowd_gt_cat = torch.cat(subject_in_crowd_gt_cat)  # triplet num, instance num, 1
                object_crowd_gt_cat = torch.cat(object_in_crowd_gt_cat)  # triplet num, instance num, 1

            subject_instance_visual_features = torch.stack(subject_instance_visual_features)
            subject_instance_location_features = torch.stack(subject_instance_location_features)

            object_instance_visual_features = torch.stack(object_instance_visual_features)
            object_instance_location_features = torch.stack(object_instance_location_features)

            if self.minus:
                predicate_feature=torch.cat([subject_language_features-object_language_features,
                                             subject_instance_visual_features-object_instance_visual_features,
                                             subject_instance_location_features-object_instance_location_features],dim=1)
            else:
                predicate_feature=torch.cat([subject_language_features,object_language_features,
                                             subject_instance_visual_features,object_instance_visual_features,
                                             subject_instance_location_features,object_instance_location_features],dim=1)

            if "crowd" in self.relation_head_list:
                if self.minus:
                    predicate_feature = torch.cat([predicate_feature,
                                                   subject_crowd_visual_features-object_crowd_visual_features,
                                                   subject_crowd_location_features-object_crowd_location_features],dim=1)
                else:
                    predicate_feature=torch.cat([predicate_feature,
                                             subject_crowd_visual_features,object_crowd_visual_features,
                                             subject_crowd_location_features, object_crowd_location_features],dim=1)
            if "phrase" in self.relation_head_list:
                predicate_feature = torch.cat([predicate_feature,
                                               phrase_visual_features],dim=1)

            predicate_feature=F.relu(self.predicate_fc1(predicate_feature))
            predicate_dist = self.predicate_fc2(predicate_feature)
            if self.reweight>0:
                if self.reweight_after:
                    predicate_out = F.sigmoid(predicate_dist) - self.reweight * self.fre_statistics.unsqueeze(0).repeat(predicate_dist.shape[0], 1)
                else:
                    predicate_out = F.sigmoid(predicate_dist - self.reweight * self.fre_statistics.unsqueeze(0).repeat(predicate_dist.shape[0], 1))
            else:
                predicate_out = F.sigmoid(predicate_dist)
            # print(predicate_out)
            if training:
                losses['loss_in_crowd_sub'] = F.binary_cross_entropy(
                                            subject_crowd_out_cat,
                                            subject_crowd_gt_cat.float(),
                                            # reduction="sum",
                                        )
                losses['loss_in_crowd_obj'] = F.binary_cross_entropy(
                                            object_crowd_out_cat,
                                            object_crowd_gt_cat.float(),
                                            # reduction="sum",
                                        )
                losses['loss_rel_pos'],losses['loss_rel_neg']=self.binary_focal_loss(predicate_out.flatten(), to_pred_phrase.gt_relation_onehots.flatten())
            else:
                to_pred_phrase.pred_subject_ids_list=to_pred_phrase.pred_subject_ids_list[:,:subject_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_object_ids_list = to_pred_phrase.pred_object_ids_list[:,:object_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_subject_in_crowds_list = subject_in_crowd_scores.squeeze(2)
                to_pred_phrase.pred_object_in_crowds_list = object_in_crowd_scores.squeeze(2)
                to_pred_phrase.pred_relation_scores=predicate_out
                results.append(to_pred_phrase)

        return results, losses, metrics

@RELATION_HEADS_REGISTRY.register()
class DifferentRelationHeads_Trans2(nn.Module):

    def __init__(self, cfg):
        super(DifferentRelationHeads_Trans2, self).__init__()

        self.device = cfg.MODEL.DEVICE
        self.instance_num = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.relation_num = cfg.MODEL.ROI_HEADS.RELATION_NUM_CLASSES
        self.relation_head_list = cfg.MODEL.RELATION_HEADS.RELATION_HEAD_LIST

        feature_types = ["language"]  # 300->512
        semantic_weights = torch.load("semantic_embedding.pth")['semantic_embedding'].to(self.device)[:80, :]
        self.semantic_embed = nn.Embedding(self.instance_num, 300)
        self.semantic_embed.load_state_dict({"weight": semantic_weights})
        self.language_fc = nn.Linear(300, 512)
        self.visual_fc = nn.Linear(1024, 512)
        self.location_fc = nn.Linear(4, 512)

        nn.init.normal_(self.language_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.language_fc.bias, 0)
        nn.init.normal_(self.visual_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.visual_fc.bias, 0)
        nn.init.normal_(self.location_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.location_fc.bias, 0)

        self.transformerEncoderLayer=CrossTransformerEncoderLayer2(d_model=512*2,nhead=8)


        instance_feature_types=["instance_visual", # 1024->256*2
                                "instance_location",] # 4->256*2

        crowd_feature_types=[]
        if "crowd" in self.relation_head_list:
            crowd_feature_types += ["crowd_visual", # 1024->512
                                    "crowd_location",] # 4->512

        phrase_feature_types = []
        if "phrase" in self.relation_head_list:
            phrase_feature_types += ["phrase_visual"] # 1024->512


        self.in_crowd_use_crowd=cfg.MODEL.RELATION_HEADS.IN_CROWD_USE_CROWD and ("crowd" in self.relation_head_list)
        self.use_attention_feature = cfg.MODEL.RELATION_HEADS.USE_ATTENTION_FEATURE
        self.attention = cfg.MODEL.RELATION_HEADS.ATTENTION
        self.minus = cfg.MODEL.RELATION_HEADS.MINUS
        self.reweight = cfg.MODEL.RELATION_HEADS.REWEIGHT
        self.reweight_after = cfg.MODEL.RELATION_HEADS.REWEIGHT_AFTER
        fre_statistics = np.load("statistics.npy", allow_pickle=True).item()
        self.fre_statistics = torch.FloatTensor([1] + fre_statistics['relation_frequency_list']).to(self.device)

        self.in_crowd_fc1 = nn.Linear(512*2,256)
        self.in_crowd_fc2 = nn.Linear(256, 1)
        nn.init.normal_(self.in_crowd_fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.in_crowd_fc1.bias, 0)
        nn.init.normal_(self.in_crowd_fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.in_crowd_fc2.bias, 0)

        if self.minus:
            self.predicate_fc1 = nn.Linear(512 * (len(feature_types)+len(instance_feature_types)+len(crowd_feature_types)+len(phrase_feature_types)), 512)
        else:
            self.predicate_fc1 = nn.Linear(512 * (2*len(feature_types)+2*len(instance_feature_types)+2*len(crowd_feature_types)+len(phrase_feature_types)), 512)
        self.predicate_ac1 = nn.ReLU()
        self.predicate_fc2 = nn.Linear(512, self.relation_num+1)
        self.predicate_ac2 = nn.Sigmoid()
        nn.init.normal_(self.predicate_fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.predicate_fc1.bias, 0)
        nn.init.normal_(self.predicate_fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.predicate_fc2.bias, 0)

    def forward_language(self,instance_classes):
        instance_class_embeddings = self.semantic_embed(instance_classes)
        instance_language_out = F.relu(self.language_fc(instance_class_embeddings))
        return instance_language_out

    def forward_visual(self,raw_visual_features):
        visual_features = self.visual_fc(raw_visual_features)
        return visual_features

    def forward_location(self,instance_boxes,phrase_boxes):
        if len(instance_boxes.shape)==3:
            phrase_union_widths = phrase_boxes[:, :, 2] - phrase_boxes[:, :, 0]
            phrase_union_heights = phrase_boxes[:, :, 3] - phrase_boxes[:, :, 1]
            instance_locations = torch.stack([(instance_boxes[:, :, 0] - phrase_boxes[:, :, 0]) / phrase_union_widths,
                                              (instance_boxes[:, :, 1] - phrase_boxes[:, :, 1]) / phrase_union_heights,
                                              (instance_boxes[:, :, 2] - phrase_boxes[:, :, 2]) / phrase_union_widths,
                                              (instance_boxes[:, :, 3] - phrase_boxes[:, :, 3]) / phrase_union_heights], dim=2)
        elif len(instance_boxes.shape)==2:
            phrase_union_widths = phrase_boxes[:, 2] - phrase_boxes[:, 0]
            phrase_union_heights = phrase_boxes[:, 3] - phrase_boxes[:, 1]
            instance_locations = torch.stack([(instance_boxes[:, 0] - phrase_boxes[:, 0]) / phrase_union_widths,
                                              (instance_boxes[:, 1] - phrase_boxes[:, 1]) / phrase_union_heights,
                                              (instance_boxes[:, 2] - phrase_boxes[:, 2]) / phrase_union_widths,
                                              (instance_boxes[:, 3] - phrase_boxes[:, 3]) / phrase_union_heights],
                                             dim=1)
        else:
            print(instance_boxes.shape)
            exit()
        location_features=self.location_fc(instance_locations)
        return location_features

    def forward_instance(self, instance_raw_visual_features,instance_boxes,phrase_boxes):
        instance_visual_features=self.forward_visual(instance_raw_visual_features)
        instance_location_features = self.forward_location(instance_boxes,phrase_boxes.unsqueeze(1).repeat(1,instance_boxes.shape[1],1))
        return instance_visual_features,instance_location_features

    def forward_crowd(self,crowd_raw_visual_features,crowd_boxes,phrase_boxes):
        crowd_visual_features = self.forward_visual(crowd_raw_visual_features)
        crowd_location_features = self.forward_location(crowd_boxes,phrase_boxes)
        return crowd_visual_features,crowd_location_features

    def forward_phrase(self,phrase_raw_visual_features):
        phrase_visual_features = self.forward_visual(phrase_raw_visual_features)
        return phrase_visual_features

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

    def forward(self, pred_instances, pred_instance_features,
                to_pred_crowds, to_pred_crowd_features,to_pred_crowd_fusion_features,
                to_pred_phrases, to_pred_phrase_features,
                training=True,iteration=1):
        losses = {}
        metrics = {}
        results = []
        for i in range(len(pred_instances)):
            pred_instance = pred_instances[i]
            pred_instance_boxes = pred_instance.pred_boxes.tensor
            pred_instance_feature = pred_instance_features[i]
            to_pred_phrase = to_pred_phrases[i]
            to_pred_phrase_boxes = to_pred_phrase.pred_boxes.tensor
            if "phrase" in self.relation_head_list:
                to_pred_phrase_feature=to_pred_phrase_features[i]
            if to_pred_crowd_features and len(to_pred_crowd_features)>0:
                pred_crowd = to_pred_crowds[i]
                pred_crowd_boxes = pred_crowd.pred_boxes.tensor
                pred_crowd_feature = to_pred_crowd_features[i]

            ## language feature for relation prediction
            subject_language_features = self.forward_language(to_pred_phrase.pred_subject_classes)
            object_language_features = self.forward_language(to_pred_phrase.pred_object_classes)

            ## phrase
            if "phrase" in self.relation_head_list:
                phrase_visual_features = self.forward_phrase(to_pred_phrase_feature)

            ## crowd
            if "crowd" in self.relation_head_list:
                subject_crowd_visual_features, subject_crowd_location_features = self.forward_crowd(
                    pred_crowd_feature[to_pred_phrase.pred_subject_crowd_ids],
                    pred_crowd_boxes[to_pred_phrase.pred_subject_crowd_ids],
                    to_pred_phrase_boxes)
                object_crowd_visual_features, object_crowd_location_features = self.forward_crowd(
                    pred_crowd_feature[to_pred_phrase.pred_object_crowd_ids],
                    pred_crowd_boxes[to_pred_phrase.pred_object_crowd_ids],
                    to_pred_phrase_boxes)

            subject_visual_features,subject_location_features=self.forward_instance(
                pred_instance_feature[to_pred_phrase.pred_subject_ids_list],
                pred_instance_boxes[to_pred_phrase.pred_subject_ids_list],
                to_pred_phrase_boxes
            )
            object_visual_features,object_location_features=self.forward_instance(
                pred_instance_feature[to_pred_phrase.pred_object_ids_list],
                pred_instance_boxes[to_pred_phrase.pred_object_ids_list],
                to_pred_phrase_boxes
            )


            encode_subject_features = self.transformerEncoderLayer(
                real_src = torch.cat([subject_visual_features,subject_location_features],dim=-1).transpose(1,0),
                gt_src = torch.cat([subject_visual_features,subject_location_features],dim=-1).transpose(1,0),
                gt_src_mask = None,
                gt_src_key_padding_mask = to_pred_phrase.pred_subject_ids_list<0,
                printsize = False, need_sigmoid_weights = False).transpose(1,0)
            encode_object_features = self.transformerEncoderLayer(
                real_src = torch.cat([object_visual_features,object_location_features],dim=-1).transpose(1,0),
                gt_src = torch.cat([object_visual_features,object_location_features],dim=-1).transpose(1,0),
                gt_src_mask = None,
                gt_src_key_padding_mask = to_pred_phrase.pred_object_ids_list<0,
                printsize = False, need_sigmoid_weights = False).transpose(1,0)

            subject_in_crowd_scores = F.sigmoid(self.in_crowd_fc2(F.relu(self.in_crowd_fc1(encode_subject_features))))
            object_in_crowd_scores = F.sigmoid(self.in_crowd_fc2(F.relu(self.in_crowd_fc1(encode_object_features))))

            subject_instance_visual_features = []
            subject_instance_location_features = []
            subject_in_crowd_score_cat = []
            subject_in_crowd_gt_cat = []
            object_instance_visual_features = []
            object_instance_location_features = []
            object_in_crowd_score_cat = []
            object_in_crowd_gt_cat = []
            for i,phrase_i in enumerate(range(len(to_pred_phrase))):
                # print(to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]])
                subject_in_crowd_scores_phrase_i=subject_in_crowd_scores[phrase_i]*to_pred_phrase.pred_subject_in_clust[phrase_i].unsqueeze(1)[:subject_in_crowd_scores[phrase_i].shape[0]]
                valid_subject_in_crowd_score = subject_in_crowd_scores_phrase_i[:to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_visual_feature = subject_visual_features[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]]
                valid_subject_location_feature = subject_location_features[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]]

                object_in_crowd_scores_phrase_i = object_in_crowd_scores[phrase_i] * to_pred_phrase.pred_object_in_clust[phrase_i].unsqueeze(1)[:object_in_crowd_scores[phrase_i].shape[0]]
                valid_object_in_crowd_score = object_in_crowd_scores_phrase_i[:to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_visual_feature = object_visual_features[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]]
                valid_object_location_feature = object_location_features[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]]

                if self.attention:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature * valid_subject_in_crowd_score, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature * valid_subject_in_crowd_score, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature * valid_object_in_crowd_score, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature * valid_object_in_crowd_score, dim=0))
                else:
                    subject_instance_visual_features.append(torch.mean(valid_subject_visual_feature, dim=0))
                    subject_instance_location_features.append(torch.mean(valid_subject_location_feature, dim=0))
                    object_instance_visual_features.append(torch.mean(valid_object_visual_feature, dim=0))
                    object_instance_location_features.append(torch.mean(valid_object_location_feature, dim=0))

                subject_in_crowd_score_cat.append(valid_subject_in_crowd_score[:, 0])
                object_in_crowd_score_cat.append(valid_object_in_crowd_score[:, 0])
                if training:
                    subject_in_crowd_gt_cat.append(to_pred_phrase.pred_subject_gt_in_crowds_list[phrase_i][:to_pred_phrase.pred_subject_lens[phrase_i]])
                    object_in_crowd_gt_cat.append(to_pred_phrase.pred_object_gt_in_crowds_list[phrase_i][:to_pred_phrase.pred_object_lens[phrase_i]])

            if training:
                for phrase_i in range(len(to_pred_phrase)):
                    if to_pred_phrase.confidence[phrase_i]==0:
                        subject_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        object_in_crowd_score_cat[phrase_i] = torch.Tensor([]).to(self.device)
                        subject_in_crowd_gt_cat[phrase_i]=torch.Tensor([]).to(self.device)
                        object_in_crowd_gt_cat[phrase_i]=torch.Tensor([]).to(self.device)

            subject_crowd_out_cat = torch.cat(subject_in_crowd_score_cat)  # triplet num, instance num, 1
            object_crowd_out_cat = torch.cat(object_in_crowd_score_cat)
            if training:
                subject_crowd_gt_cat = torch.cat(subject_in_crowd_gt_cat)  # triplet num, instance num, 1
                object_crowd_gt_cat = torch.cat(object_in_crowd_gt_cat)  # triplet num, instance num, 1

            subject_instance_visual_features = torch.stack(subject_instance_visual_features)
            subject_instance_location_features = torch.stack(subject_instance_location_features)

            object_instance_visual_features = torch.stack(object_instance_visual_features)
            object_instance_location_features = torch.stack(object_instance_location_features)

            if self.minus:
                predicate_feature=torch.cat([subject_language_features-object_language_features,
                                             subject_instance_visual_features-object_instance_visual_features,
                                             subject_instance_location_features-object_instance_location_features],dim=1)
            else:
                predicate_feature=torch.cat([subject_language_features,object_language_features,
                                             subject_instance_visual_features,object_instance_visual_features,
                                             subject_instance_location_features,object_instance_location_features],dim=1)

            if "crowd" in self.relation_head_list:
                if self.minus:
                    predicate_feature = torch.cat([predicate_feature,
                                                   subject_crowd_visual_features-object_crowd_visual_features,
                                                   subject_crowd_location_features-object_crowd_location_features],dim=1)
                else:
                    predicate_feature=torch.cat([predicate_feature,
                                             subject_crowd_visual_features,object_crowd_visual_features,
                                             subject_crowd_location_features, object_crowd_location_features],dim=1)
            if "phrase" in self.relation_head_list:
                predicate_feature = torch.cat([predicate_feature,
                                               phrase_visual_features],dim=1)

            predicate_feature=self.predicate_ac1(self.predicate_fc1(predicate_feature))
            predicate_dist = self.predicate_fc2(predicate_feature)

            if self.reweight>0:
                if self.reweight_after:
                    predicate_out = self.predicate_ac2(predicate_dist) - self.reweight * self.fre_statistics.unsqueeze(0).repeat(predicate_dist.shape[0], 1)
                else:
                    predicate_out = self.predicate_ac2(predicate_dist - self.reweight * self.fre_statistics.unsqueeze(0).repeat(predicate_dist.shape[0], 1))
            else:
                predicate_out = self.predicate_ac2(predicate_dist)

            if training:
                losses['loss_in_crowd_sub'] = F.binary_cross_entropy(
                                            subject_crowd_out_cat,
                                            subject_crowd_gt_cat.float(),
                                            # reduction="sum",
                                        )
                losses['loss_in_crowd_obj'] = F.binary_cross_entropy(
                                            object_crowd_out_cat,
                                            object_crowd_gt_cat.float(),
                                            # reduction="sum",
                                        )
                losses['loss_rel_pos'],losses['loss_rel_neg']=self.binary_focal_loss(predicate_out.flatten(), to_pred_phrase.gt_relation_onehots.flatten())
            else:
                to_pred_phrase.pred_subject_ids_list=to_pred_phrase.pred_subject_ids_list[:,:subject_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_object_ids_list = to_pred_phrase.pred_object_ids_list[:,:object_in_crowd_scores.shape[1]]
                to_pred_phrase.pred_subject_in_crowds_list = subject_in_crowd_scores.squeeze(2)
                to_pred_phrase.pred_object_in_crowds_list = object_in_crowd_scores.squeeze(2)
                to_pred_phrase.pred_relation_scores=predicate_out
                results.append(to_pred_phrase)

        return results, losses, metrics

