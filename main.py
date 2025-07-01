# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import time
import json
import h5py
import logging
import os
import cv2
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from pycocotools import mask as maskUtils
from matplotlib import pyplot as plt
from collections import defaultdict
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from detectron2.engine import default_argument_parser, default_setup, launch
args = default_argument_parser().parse_args()
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
# torch.set_deterministic(True)

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor, MaskPredictor
from detectron2.modeling import build_model
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from panopticapi.utils import IdGenerator
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.common import MapDataset, DatasetFromList

logger = logging.getLogger("detectron2")

def do_crowd_train(cfg, model, resume=False,print_size=True):
    model.eval()
    for param in model.named_parameters():
        param[1].requires_grad = False
    for param in model.named_parameters():
        for trainable in cfg.MODEL.TRAINABLE:
            if param[0].startswith(trainable):
                param[1].requires_grad = True
                break

    saved_model=torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device("cpu"))
    last_iter=saved_model.get("iteration", -1)
    last_iter = -1
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer,last_iter)
    metrics_sum_dict = {
        'example': 0,
    }
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler, metrics_sum_dict=metrics_sum_dict
    )
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    start_iter = last_iter+1

    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    metrics_pr_dict={}
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    acumulate_losses=0
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()
            # if True:
            try:
                results_dict, losses_dict, metrics_dict = model(data, iteration, mode="crowd", training=True,print_size=print_size)

                # instances = results_dict[0]['gt_instances']
                # crowds = results_dict[0]['gt_crowds']
                # visualizer = Visualizer(read_image(data[0]['file_name'], format="BGR")[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]),instance_mode=ColorMode.IMAGE)
                # vis_output_instance = visualizer.draw_instance_predictions(instances.to('cpu'))
                # vis_output_instance.save(os.path.join(cfg.OUTPUT_DIR,str(data[0]['image_id'])+"_instance.png"))
                # visualizer = Visualizer(read_image(data[0]['file_name'], format="BGR")[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), instance_mode=ColorMode.IMAGE)
                # vis_output_crowd = visualizer.draw_instance_predictions(crowds.to('cpu'))
                # vis_output_crowd.save(os.path.join(cfg.OUTPUT_DIR, str(data[0]['image_id']) + "_crowd.png"))
                # exit()
                if len(losses_dict)==0:
                    continue
                losses = sum(loss for loss in losses_dict.values())
                # print(losses,losses_dict)
                assert torch.isfinite(losses).all(), losses_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(losses_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                acumulate_losses += losses_reduced
                if comm.is_main_process():
                    storage.put_scalars(acumulate_losses=acumulate_losses/(iteration-start_iter),total_loss=losses_reduced, **loss_dict_reduced)

                storage.put_scalars(**metrics_dict, smoothing_hint=False)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)

def do_phrase_train(cfg, model, resume=False,print_size=True):
    model.eval()
    for param in model.named_parameters():
        param[1].requires_grad = False
    for param in model.named_parameters():
        for trainable in cfg.MODEL.TRAINABLE:
            if param[0].startswith(trainable):
                param[1].requires_grad = True
                break

    saved_model = torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device("cpu"))
    last_iter = saved_model.get("iteration", -1)
    last_iter = -1
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer, last_iter)
    metrics_sum_dict = {
        'example': 0,
    }
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler, metrics_sum_dict=metrics_sum_dict
    )
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    start_iter = last_iter + 1
    # state_dict=torch.load(cfg.MODEL.WEIGHTS).pop("model")
    # model.load_state_dict(state_dict,strict=False)
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    metrics_pr_dict={}
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    acumulate_losses=0
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            # print(iteration)
            iteration = iteration + 1
            storage.step()
            if True:
            # try:
                results_dict, losses_dict, metrics_dict = model(data, iteration, mode="phrase", training=True,print_size=print_size)
                # print(losses_dict)
                if len(losses_dict)==0:
                    continue
                losses = sum(loss for loss in losses_dict.values())
                assert torch.isfinite(losses).all(), losses_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(losses_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                acumulate_losses += losses_reduced
                if comm.is_main_process():
                    storage.put_scalars(acumulate_losses=acumulate_losses/(iteration-start_iter),total_loss=losses_reduced, **loss_dict_reduced)

                storage.put_scalars(**metrics_dict, smoothing_hint=False)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)
                torch.cuda.empty_cache()
            # except Exception as e:
            #     print(e)

def do_relation_train(cfg, model, resume=False,print_size=True):
    model.train()
    for param in model.named_parameters():
        param[1].requires_grad = False
    for param in model.named_parameters():
        for trainable in cfg.MODEL.TRAINABLE:
            if param[0].startswith(trainable):
                param[1].requires_grad = True
                break

        if param[0] == "relation_heads.semantic_embed.weight" or param[0] == "relation_heads.fre_statistics":
            param[1].requires_grad = False

    metrics_sum_dict = {
        'example': 0,
    }

    last_iter = -1
    # try:
    #     saved_model=torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device("cpu"))
    #     last_iter=saved_model.get("iteration", -1)
    # except Exception as e:
    #     print(e)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer,last_iter)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler, metrics_sum_dict=metrics_sum_dict
    )
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    start_iter = last_iter+1

    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    metrics_pr_dict={}
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    acumulate_losses=0
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            # iteration = iteration + 1
            storage.step()
            if True:
            # try:
                results_dict, losses_dict, metrics_dict = model(data, iteration, mode="relation", training=True,print_size=print_size)
                # print(losses_dict)
                if print_size:
                    print("=======================================================")
                iteration = iteration + 1
                if len(losses_dict)>0:
                    losses = sum(loss for loss in losses_dict.values())
                    assert torch.isfinite(losses).all(), losses_dict

                    loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(losses_dict).items()}
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    acumulate_losses += losses_reduced
                    if comm.is_main_process():
                        storage.put_scalars(acumulate_losses=acumulate_losses/(iteration-start_iter),total_loss=losses_reduced, **loss_dict_reduced)

                    if 'relation_cls_tp' in metrics_dict:
                        metrics_sum_dict['relation_cls_tp_sum']+=metrics_dict['relation_cls_tp']
                        metrics_sum_dict['relation_cls_p_sum'] += metrics_dict['relation_cls_p']
                        metrics_pr_dict['relation_cls_precision'] = metrics_sum_dict['relation_cls_tp_sum'] / metrics_sum_dict['relation_cls_p_sum']
                    if 'triplet_tp' in metrics_dict:
                        metrics_sum_dict['triplet_tp_sum'] += metrics_dict['triplet_tp']
                        metrics_sum_dict['triplet_tp20_sum'] += metrics_dict['triplet_tp20']
                        metrics_sum_dict['triplet_tp50_sum'] += metrics_dict['triplet_tp50']
                        metrics_sum_dict['triplet_tp100_sum'] += metrics_dict['triplet_tp100']
                        metrics_sum_dict['triplet_p_sum'] += metrics_dict['triplet_p']
                        metrics_sum_dict['triplet_p20_sum'] += metrics_dict['triplet_p20']
                        metrics_sum_dict['triplet_p50_sum'] += metrics_dict['triplet_p50']
                        metrics_sum_dict['triplet_p100_sum'] += metrics_dict['triplet_p100']
                        metrics_sum_dict['triplet_g_sum'] += metrics_dict['triplet_g']
                        metrics_pr_dict['triplet_precision'] = metrics_sum_dict['triplet_tp_sum'] / metrics_sum_dict['triplet_p_sum']
                        metrics_pr_dict['triplet_precision20'] = metrics_sum_dict['triplet_tp20_sum'] / metrics_sum_dict['triplet_p20_sum']
                        metrics_pr_dict['triplet_precision50'] = metrics_sum_dict['triplet_tp50_sum'] / metrics_sum_dict['triplet_p50_sum']
                        metrics_pr_dict['triplet_precision100'] = metrics_sum_dict['triplet_tp100_sum'] / metrics_sum_dict['triplet_p100_sum']
                        metrics_pr_dict['triplet_recall'] = metrics_sum_dict['triplet_tp_sum'] / metrics_sum_dict['triplet_g_sum']
                        metrics_pr_dict['triplet_recall20'] = metrics_sum_dict['triplet_tp20_sum'] / metrics_sum_dict['triplet_g_sum']
                        metrics_pr_dict['triplet_recall50'] = metrics_sum_dict['triplet_tp50_sum'] / metrics_sum_dict['triplet_g_sum']
                        metrics_pr_dict['triplet_recall100'] = metrics_sum_dict['triplet_tp100_sum'] / metrics_sum_dict['triplet_g_sum']

                    storage.put_scalars(**metrics_pr_dict, smoothing_hint=False)

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                    scheduler.step()

                if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)
                torch.cuda.empty_cache()
            # except Exception as e:
            #     print(e)
            #     torch.cuda.empty_cache()

def do_relation_test(cfg, model, visible=False,print_size=True):
    print(cfg.OUTPUT_DIR)
    image_dict = {}
    image_triplet_dict = {}
    less=""
    if cfg.LESS:
        less="_less"
    if not visible:
        result_f=h5py.File(os.path.join(cfg.OUTPUT_DIR,"result"+less+".h5"),'w')
    dataset_name = cfg.DATASETS.TEST[0]
    test_images_dict = json.load(open(MetadataCatalog.get(dataset_name).get("instance_json_file"), 'r'))
    image_path = MetadataCatalog.get(dataset_name).get("image_path")

    predictor = DefaultPredictor(cfg)

    test_image_ids_list=list(test_images_dict.keys())
    total = len(test_images_dict)
    count = 0
    # for idx, inputs in enumerate(data_loader):
    processbar=tqdm(total=len(test_image_ids_list),unit="image")
    for image_i,image_id in enumerate(test_image_ids_list):
        if image_i == "373697":
            continue
        image_info = test_images_dict[image_id]
        img = read_image(image_path + "/" + image_info['image_name'], format="BGR")
        count += 1

        if True:
        # try:
            predictions = predictor(img, 0, mode="relation",print_size=print_size)
            if len(predictions)==0 or 'phrases' not in predictions[0]:
                print(image_id)
                continue
            instances=predictions[0]['instances']
            phrases = predictions[0]['phrases']
            # print(instances.pred_classes)
            # print(phrases.pred_subject_ids_list)
            # print(phrases.pred_object_ids_list)
            # print(phrases.pred_subject_in_crowds_list)
            # print(phrases.pred_object_in_crowds_list)
            # exit()
            instance_pred_classes=instances.pred_classes.data.cpu().numpy().tolist()
            instance_pred_boxes=instances.pred_boxes.tensor.data.cpu().numpy().tolist()
            instance_pred_masks=instances.pred_masks.data.cpu().numpy()
            instances_dict={}
            for i in range(len(instances)):
                seg=maskUtils.encode(np.asfortranarray(instance_pred_masks[i]))
                instances_dict[str(i)]={
                    "instance_id":i,
                    "instance_class_id":instance_pred_classes[i]+1,
                    "box":[instance_pred_boxes[i][0],instance_pred_boxes[i][1],instance_pred_boxes[i][2]-instance_pred_boxes[i][0],instance_pred_boxes[i][3]-instance_pred_boxes[i][1]],
                    "segmentation": {'counts':seg['counts'].decode(),'size':seg['size']}
                }

            image_dict[image_id]={
                                    "image_id":image_info['image_id'],
                                    "height":image_info['height'],
                                    "width":image_info['width'],
                                    "image_name":image_info['image_name'],
                                    "instances": instances_dict
                                  }
            if not visible:
                result_f[image_id + "_pred_subject_in_crowds_list"] = phrases.pred_subject_in_crowds_list.data.cpu().numpy()
                result_f[image_id + "_pred_object_in_crowds_list"] = phrases.pred_object_in_crowds_list.data.cpu().numpy()
                result_f[image_id + "_pred_subject_ids_list"] = phrases.pred_subject_ids_list.data.cpu().numpy()
                result_f[image_id + "_pred_object_ids_list"] = phrases.pred_object_ids_list.data.cpu().numpy()
                result_f[image_id + "_pred_relation_scores"] = phrases.pred_relation_scores.data.cpu().numpy()

            image_triplet_dict[image_id]={
                "image_id": image_info['image_id'],
                "height": image_info['height'],
                "width": image_info['width'],
                "image_name": image_info['image_name'],
                "triplets": []#triplets_dict
            }
            torch.cuda.empty_cache()
        # except Exception as e:
        #     print(e)
        #     torch.cuda.empty_cache()
        processbar.update(1)
    processbar.close()

    if not visible:
        json.dump(image_dict, open(os.path.join(cfg.OUTPUT_DIR, "test_images_dict"+less+".json"), 'w'))
        json.dump(image_triplet_dict, open(os.path.join(cfg.OUTPUT_DIR, "test_images_triplets_dict"+less+".json"), 'w'))
        result_f.close()
    return image_dict,image_triplet_dict

def generate_triplets(cfg, in_crowd_threshold,visible,merge):
    print(cfg.OUTPUT_DIR)
    less=""
    if cfg.LESS:
        less="_less"
    image_dict=json.load(open(os.path.join(cfg.OUTPUT_DIR,"test_images_dict"+less+".json"),'r'))
    image_triplet_dict=json.load(open(os.path.join(cfg.OUTPUT_DIR,"test_images_triplets_dict"+less+".json"),'r'))
    dataset_name = cfg.DATASETS.TEST[0]
    image_path = MetadataCatalog.get(dataset_name).get("image_path")

    gt_triplet_dict = json.load(open("../data/gvrd/test/images_triplets_dict.json"))
    gt_image_dict = json.load(open("../data/gvrd/test/images_dict.json"))

    result_f=h5py.File(os.path.join(cfg.OUTPUT_DIR,"result"+less+".h5"),'r')
    total=len(image_triplet_dict)
    count=0
    valid_vis_images_ids=[]
    image_count=defaultdict(list)
    matched_valid_vis_images_ids=[]
    processbar=tqdm(total=len(image_triplet_dict),unit="image")
    for image_id in image_triplet_dict:
        count += 1
        print(image_id, str(count) + "/" + str(total))

        image_count[image_id]=[]
        pred_image_instances = image_dict[image_id]['instances']
        gt_triplets = gt_triplet_dict[image_id]['triplets']
        gt_image_instances = gt_image_dict[image_id]['instances']
        gt_img_h,gt_img_w=gt_image_dict[image_id]['height'],gt_image_dict[image_id]['width']
        cls_dict=defaultdict(int)
        for instance_id in pred_image_instances:
            cls_dict[pred_image_instances[instance_id]['instance_class_id']]+=1

        phrases={}
        phrases['pred_subject_in_crowds_list'] = torch.Tensor(result_f[image_id + "_pred_subject_in_crowds_list"])#.cuda()
        phrases['pred_object_in_crowds_list'] = torch.Tensor(result_f[image_id + "_pred_object_in_crowds_list"])#.cuda()
        phrases['pred_subject_ids_list'] = torch.Tensor(result_f[image_id + "_pred_subject_ids_list"]).long()#.cuda()
        phrases['pred_object_ids_list'] = torch.Tensor(result_f[image_id + "_pred_object_ids_list"]).long()#.cuda()
        phrases['pred_relation_scores'] = torch.Tensor(result_f[image_id + "_pred_relation_scores"])#.cuda()

        # print(phrases['pred_subject_ids_list'])
        # print(phrases['pred_subject_in_crowds_list']*(phrases['pred_subject_ids_list']>=0).float())
        # print(phrases['pred_object_ids_list'])
        # print(phrases['pred_object_in_crowds_list']*(phrases['pred_object_ids_list']>=0).float())
        # print(phrases['pred_relation_scores'])
        # print("-----------------------------------")
        # exit()

        phrase_pred_subject_in_crowds_list = phrases['pred_subject_in_crowds_list']
        phrase_pred_subject_in_crowds_list[((phrase_pred_subject_in_crowds_list < in_crowd_threshold) + (phrases['pred_subject_ids_list'] == -1)) > 0] = 0
        phrase_pred_object_in_crowds_list = phrases['pred_object_in_crowds_list']
        phrase_pred_object_in_crowds_list[((phrase_pred_object_in_crowds_list < in_crowd_threshold) + (phrases['pred_object_ids_list'] == -1)) > 0] = 0

        phrase_select = torch.where((torch.sum(phrase_pred_subject_in_crowds_list, dim=1) > 0) * (torch.sum(phrase_pred_object_in_crowds_list, dim=1) > 0) > 0)  # valid phrase
        phrase_pred_relation_scores = phrases['pred_relation_scores'][:, 1:][phrase_select]
        phrase_pred_subject_ids_list = phrases['pred_subject_ids_list'][phrase_select]
        phrase_pred_object_ids_list = phrases['pred_object_ids_list'][phrase_select]
        phrase_pred_subject_in_crowds_list = phrase_pred_subject_in_crowds_list[phrase_select]
        phrase_pred_object_in_crowds_list = phrase_pred_object_in_crowds_list[phrase_select]

        topk_scores, topk_flatten_indx = torch.sort(phrase_pred_relation_scores.flatten(), descending=True)
        topk_phrase_ids = topk_flatten_indx // 96
        topk_relation_ids = topk_flatten_indx % 96
        subject_ids_list = []
        object_ids_list = []
        subject_scores_list = []
        object_scores_list = []
        relation_ids = []
        scores = []
        triplet_ids_list=[]
        for i in range(topk_scores.shape[0]):
            if len(triplet_ids_list)>=100:
                break
            phrase_id = topk_phrase_ids[i].item()
            subject_ids = phrase_pred_subject_ids_list[phrase_id]
            object_ids = phrase_pred_object_ids_list[phrase_id]
            relation_id = topk_relation_ids[i].item()
            topk_score = topk_scores[i].item()

            subject_in_crowd_scores = phrase_pred_subject_in_crowds_list[phrase_id]
            object_in_crowd_scores = phrase_pred_object_in_crowds_list[phrase_id]
            # print(subject_in_crowd_scores[subject_ids>0])
            # print(object_in_crowd_scores[object_ids>0])
            # print(subject_ids, subject_in_crowd_scores)
            # print(object_ids, object_in_crowd_scores)
            # print(topk_score)
            subject_idxs = torch.where(subject_in_crowd_scores > 0)[0]
            object_idxs = torch.where(object_in_crowd_scores > 0)[0]

            if subject_idxs.shape[0] > 0 and object_idxs.shape[0] > 0:
                single_subject_scores=subject_in_crowd_scores[subject_idxs].data.cpu().numpy().tolist()
                single_object_scores=object_in_crowd_scores[object_idxs].data.cpu().numpy().tolist()
                single_subject_ids=subject_ids[subject_idxs].data.cpu().numpy().tolist()
                single_object_ids=object_ids[object_idxs].data.cpu().numpy().tolist()
                notpostprocess=False
                if notpostprocess:
                    single_subject_ids_only=single_subject_ids
                    single_object_ids_only = single_object_ids
                    triplet_id = str(sorted(single_subject_ids_only)) + "_" + str(sorted(single_object_ids_only)) + "_" + str(relation_id)
                    triplet_ids_list.append(triplet_id)
                    subject_ids_list.append(single_subject_ids_only)
                    object_ids_list.append(single_object_ids_only)
                    subject_scores_list.append(single_subject_scores)
                    object_scores_list.append(single_object_scores)
                    relation_ids.append(relation_id)
                    scores.append(topk_score)
                else: ## TODO subject_scores_list and object_scores_list
                    single_subject_ids_only=list(set(single_subject_ids).difference(set(single_object_ids)))
                    single_object_ids_only=list(set(single_object_ids).difference(set(single_subject_ids)))
                    single_ids_intersection=set(single_subject_ids).intersection(set(single_object_ids))
                    for idx in single_ids_intersection:
                        suj_score=single_subject_scores[single_subject_ids.index(idx)]
                        obj_score=single_object_scores[single_object_ids.index(idx)]
                        if suj_score >= obj_score:
                            single_subject_ids_only.append(idx)
                        elif suj_score < obj_score:
                            single_object_ids_only.append(idx)
                    triplet_id=str(sorted(single_subject_ids_only))+"_"+str(sorted(single_object_ids_only))+"_"+str(relation_id)
                    if len(single_subject_ids_only)==0 \
                            or len(single_object_ids_only)==0 \
                            or triplet_id in triplet_ids_list \
                            or (len(single_subject_ids_only)==1 and len(single_object_ids_only)==1):
                        continue
                    replace=False
                    if merge=="merge":
                        for exist_i,(exist_triplet_ids,exist_subject_ids,exist_object_ids,exist_relation_id,exist_relation_id) in enumerate(zip(triplet_ids_list,subject_ids_list,object_ids_list,relation_ids,relation_ids)):
                            if exist_relation_id==relation_id and \
                                len(set(single_subject_ids_only).difference(set(exist_subject_ids)))>=0 \
                                and len(set(exist_subject_ids).intersection(set(single_subject_ids_only)))>0 \
                                and len(set(single_object_ids_only).difference(set(exist_object_ids)))>=0 \
                                and len(set(exist_object_ids).intersection(set(single_object_ids_only)))>0:
                                single_subject_ids_only=list(set(single_subject_ids_only).union(set(exist_subject_ids)))
                                single_object_ids_only = list(set(single_object_ids_only).union(set(exist_object_ids)))
                                triplet_id=str(sorted(single_subject_ids_only))+"_"+str(sorted(single_object_ids_only))+"_"+str(relation_id)
                                triplet_ids_list[exist_i]=triplet_id
                                subject_ids_list[exist_i]=single_subject_ids_only
                                object_ids_list[exist_i]=single_object_ids_only
                                replace=True
                                break
                    elif merge=="skip":
                        for exist_i,(exist_triplet_ids,exist_subject_ids,exist_object_ids,exist_relation_id,exist_relation_id) in enumerate(zip(triplet_ids_list,subject_ids_list,object_ids_list,relation_ids,relation_ids)):
                            if exist_relation_id==relation_id and \
                                len(set(single_subject_ids_only).difference(set(exist_subject_ids)))>=0 \
                                and len(set(exist_subject_ids).intersection(set(single_subject_ids_only)))>0 \
                                and len(set(single_object_ids_only).difference(set(exist_object_ids)))>=0 \
                                and len(set(exist_object_ids).intersection(set(single_object_ids_only)))>0:
                                # single_subject_ids_only=list(set(single_subject_ids_only).union(set(exist_subject_ids)))
                                # single_object_ids_only = list(set(single_object_ids_only).union(set(exist_object_ids)))
                                # triplet_id=str(sorted(single_subject_ids_only))+"_"+str(sorted(single_object_ids_only))+"_"+str(relation_id)
                                # triplet_ids_list[exist_i]=triplet_id
                                # subject_ids_list[exist_i]=single_subject_ids_only
                                # object_ids_list[exist_i]=single_object_ids_only
                                replace=True
                                break
                    if not replace:
                        triplet_ids_list.append(triplet_id)
                        subject_ids_list.append(single_subject_ids_only)
                        object_ids_list.append(single_object_ids_only)
                        relation_ids.append(relation_id)
                        scores.append(topk_score)
                # for visualization and statistics
                if len(triplet_ids_list) <= 5:# and (len(subject_idxs) < torch.sum(subject_ids>=0).item() or len(object_idxs) < torch.sum(object_ids>=0).item()):
                    # print(image_id,i)
                    # print(subject_idxs)
                    # print(subject_ids)
                    # print(object_idxs)
                    # print(object_ids)
                    if visible:
                        image_info=image_dict[image_id]
                        img = read_image(image_path+"/" + image_id.zfill(12)+".jpg", format="RGB")

                        triplets_subject_classes = []
                        triplets_object_classes = []
                        triplets_subject_scores = single_subject_scores if notpostprocess else None
                        triplets_object_scores = single_object_scores if notpostprocess else None
                        triplets_subject_boxes = []
                        triplets_object_boxes = []
                        triplets_subject_masks = []
                        triplets_object_masks = []
                        for sub in single_subject_ids_only:
                            subject_instance = image_info['instances'][str(sub)]
                            triplets_subject_classes.append(subject_instance['instance_class_id'])
                            triplets_subject_boxes.append([subject_instance['box'][0], subject_instance['box'][1],
                                                           subject_instance['box'][2] + subject_instance['box'][0],
                                                           subject_instance['box'][3] + subject_instance['box'][1]])
                            triplets_subject_masks.append(subject_instance['segmentation'])
                        for obj in single_object_ids_only:
                            object_instance = image_info['instances'][str(obj)]
                            triplets_object_classes.append(object_instance['instance_class_id'])
                            triplets_object_boxes.append([object_instance['box'][0], object_instance['box'][1],
                                                          object_instance['box'][2] + object_instance['box'][0],
                                                          object_instance['box'][3] + object_instance['box'][1]])
                            triplets_object_masks.append(object_instance['segmentation'])

                        triplets_subject_classes = torch.Tensor(triplets_subject_classes).int() - 1
                        triplets_object_classes = torch.Tensor(triplets_object_classes).int() - 1
                        triplets_subject_boxes = Boxes(torch.Tensor(triplets_subject_boxes))
                        triplets_object_boxes = Boxes(torch.Tensor(triplets_object_boxes))
                        triplets_subject_masks = triplets_subject_masks
                        triplets_object_masks = triplets_object_masks
                        triplets_relations = torch.Tensor([relation_id]).int()
                        triplets_scores = torch.Tensor([topk_score])

                        visualizer = Visualizer(img, MetadataCatalog.get(dataset_name), instance_mode=ColorMode.IMAGE)
                        vis_output_instance, relation = visualizer.draw_group_relation_predictions(image_info['height'],
                                                                                                   image_info['width'],
                                                                                                   triplets_subject_classes,
                                                                                                   triplets_object_classes,
                                                                                                   triplets_subject_scores,
                                                                                                   triplets_object_scores,
                                                                                                   triplets_subject_boxes,
                                                                                                   triplets_object_boxes,
                                                                                                   triplets_subject_masks,
                                                                                                   triplets_object_masks,
                                                                                                   triplets_relations,
                                                                                                   triplets_scores)
                        vis_output_instance.save(os.path.join("vis",image_id+"_"+str(i)+".png"))
                        # exit()
                    if image_id not in valid_vis_images_ids:
                        valid_vis_images_ids.append(image_id)

        triplets_dict = {}
        # print(phrase_select[0].shape[0],len(scores))
        if len(scores) > 0:
            matched_gt=[]
            for i, (subject_ids, object_ids, relation_id, score) in enumerate(
                    zip(subject_ids_list, object_ids_list, relation_ids, scores)):
                triplets_dict[str(i)] = {
                    "triplet_id": i,
                    "subject_instance": subject_ids,
                    "object_instance": object_ids,
                    "relation_id": relation_id + 1,
                    "score": score
                }
                subject_cls=pred_image_instances[str(subject_ids[0])]['instance_class_id']
                object_cls = pred_image_instances[str(object_ids[0])]['instance_class_id']
                matched=False
                for k in gt_triplets:
                    if k in matched_gt:
                        continue
                    if (pred_image_instances[str(subject_ids[0])]['instance_class_id'], relation_id+1, pred_image_instances[str(object_ids[0])]['instance_class_id']) \
                            != (gt_image_instances[str(gt_triplets[k]['subject_instance'][0])]['instance_class_id'], gt_triplets[k]['relation_id'], gt_image_instances[str(gt_triplets[k]['object_instance'][0])]['instance_class_id']):
                        continue

                    gt_subj_crowd_mask = union_boxes_to_mask([gt_image_instances[str(gt_ins_id)]['box'] for gt_ins_id in gt_triplets[k]['subject_instance']], gt_img_h, gt_img_w)
                    gt_obj_crowd_mask = union_boxes_to_mask([gt_image_instances[str(gt_ins_id)]['box'] for gt_ins_id in gt_triplets[k]['object_instance']], gt_img_h, gt_img_w)
                    pred_subj_crowd_mask = union_boxes_to_mask([pred_image_instances[str(pred_ins_id)]['box'] for pred_ins_id in subject_ids], gt_img_h, gt_img_w)
                    pred_obj_crowd_mask = union_boxes_to_mask([pred_image_instances[str(pred_ins_id)]['box'] for pred_ins_id in object_ids], gt_img_h, gt_img_w)

                    pred_subj_crowd_mask = cv2.resize(pred_subj_crowd_mask, (gt_img_w, gt_img_h), cv2.INTER_NEAREST)
                    pred_obj_crowd_mask = cv2.resize(pred_obj_crowd_mask, (gt_img_w, gt_img_h), cv2.INTER_NEAREST)

                    subj_iou = compute_pixel_iou(pred_subj_crowd_mask, gt_subj_crowd_mask)
                    obj_iou = compute_pixel_iou(pred_obj_crowd_mask, gt_obj_crowd_mask)
                    # 这里的匹配是iou超过阈值就可以了，并没有找最匹配的，因为可能subj_mask是subj_iou_max，但是obj_mask不是, 而max的subj和obj之间又没有关系
                    if subj_iou > 0.5 and obj_iou > 0.5:
                        matched_gt.append(k)
                        matched=True
                        image_count[image_id].append([subject_ids,object_ids,relation_id+1,score])
                        if len(subject_ids) < cls_dict[subject_cls] or len(object_ids) < cls_dict[object_cls]:
                            if image_id not in matched_valid_vis_images_ids:
                                matched_valid_vis_images_ids.append(image_id)

        image_triplet_dict[image_id]['triplets']=triplets_dict
        processbar.update(1)
    processbar.close()
    result_f.close()
    if len(merge)>0:
        json.dump(image_triplet_dict, open(os.path.join(cfg.OUTPUT_DIR, "test_images_triplets_dict"+str(in_crowd_threshold)+less+"_"+merge+".json"), 'w'))
        json.dump(valid_vis_images_ids,open(os.path.join(cfg.OUTPUT_DIR, "valid_vis_image_ids_"+str(in_crowd_threshold)+less+"_"+merge+".json"), 'w'))
    else:
        json.dump(image_triplet_dict, open(os.path.join(cfg.OUTPUT_DIR, "test_images_triplets_dict"+str(in_crowd_threshold)+less+".json"), 'w'))
        json.dump(valid_vis_images_ids,open(os.path.join(cfg.OUTPUT_DIR, "valid_vis_image_ids_"+str(in_crowd_threshold)+less+".json"), 'w'))
    json.dump(image_count,open(os.path.join(cfg.OUTPUT_DIR, "test"+str(in_crowd_threshold)+".json"),'w'))
    json.dump(matched_valid_vis_images_ids,open(os.path.join(cfg.OUTPUT_DIR, "valid100.json"),'w'))

def generate_triplets_by_predicate(cfg, in_crowd_threshold,visible,merge):
    print(cfg.OUTPUT_DIR)
    less=""
    if cfg.LESS:
        less="_less"
    image_dict=json.load(open(os.path.join(cfg.OUTPUT_DIR,"test_images_dict"+less+".json"),'r'))
    image_triplet_dict=json.load(open(os.path.join(cfg.OUTPUT_DIR,"test_images_triplets_dict"+less+".json"),'r'))
    dataset_name = cfg.DATASETS.TEST[0]
    image_path = MetadataCatalog.get(dataset_name).get("image_path")

    gt_triplet_dict = json.load(open("../data/gvrd/test/images_triplets_dict.json"))
    gt_image_dict = json.load(open("../data/gvrd/test/images_dict.json"))

    result_f=h5py.File(os.path.join(cfg.OUTPUT_DIR,"result"+less+".h5"),'r')
    total=len(image_triplet_dict)
    count=0
    valid_vis_images_ids=[]
    image_count=defaultdict(list)
    matched_valid_vis_images_ids=[]
    processbar=tqdm(total=len(image_triplet_dict),unit="image")
    for image_id in image_triplet_dict:
        count += 1
        # print(str(count) + "/" + str(total))

        image_count[image_id]=[]
        pred_image_instances = image_dict[image_id]['instances']
        gt_triplets = gt_triplet_dict[image_id]['triplets']
        gt_image_instances = gt_image_dict[image_id]['instances']
        gt_img_h,gt_img_w=gt_image_dict[image_id]['height'],gt_image_dict[image_id]['width']
        cls_dict=defaultdict(int)
        for instance_id in pred_image_instances:
            cls_dict[pred_image_instances[instance_id]['instance_class_id']]+=1

        phrases={}
        phrases['pred_subject_in_crowds_list'] = torch.Tensor(result_f[image_id + "_pred_subject_in_crowds_list"])#.cuda()
        phrases['pred_object_in_crowds_list'] = torch.Tensor(result_f[image_id + "_pred_object_in_crowds_list"])#.cuda()
        phrases['pred_subject_ids_list'] = torch.Tensor(result_f[image_id + "_pred_subject_ids_list"]).long()#.cuda()
        phrases['pred_object_ids_list'] = torch.Tensor(result_f[image_id + "_pred_object_ids_list"]).long()#.cuda()
        phrases['pred_relation_scores'] = torch.Tensor(result_f[image_id + "_pred_relation_scores"])#.cuda()

        flatten_phrase_pred_subject_ids_list=phrases['pred_subject_ids_list'].unsqueeze(-1).repeat(1,1,96+1).transpose(1,0).flatten(0,1) # phrase*97,max_len
        flatten_phrase_pred_object_ids_list = phrases['pred_object_ids_list'].unsqueeze(-1).repeat(1, 1, 96 + 1).transpose(1,0).flatten(0,1) # phrase*97,max_len
        flatten_phrase_pred_subject_in_crowds_list=phrases['pred_subject_in_crowds_list'].transpose(1,2).flatten(0,1)
        flatten_phrase_pred_object_in_crowds_list = phrases['pred_object_in_crowds_list'].transpose(1, 2).flatten(0,1)
        flatten_pred_relation_scores=phrases['pred_relation_scores'][:,1:].flatten()
        flatten_phrase_relation_ids_list = torch.arange(0, 96).unsqueeze(0).repeat(phrases['pred_relation_scores'].shape[0], 1).flatten()

        flatten_phrase_pred_subject_in_crowds_list[((flatten_phrase_pred_subject_in_crowds_list < in_crowd_threshold) + (flatten_phrase_pred_subject_ids_list == -1)) > 0] = 0
        flatten_phrase_pred_object_in_crowds_list[((flatten_phrase_pred_object_in_crowds_list < in_crowd_threshold) + (flatten_phrase_pred_object_ids_list == -1)) > 0] = 0

        phrase_select = torch.where((torch.sum(flatten_phrase_pred_subject_in_crowds_list, dim=1) > 0) * (torch.sum(flatten_phrase_pred_object_in_crowds_list, dim=1) > 0) > 0)  # valid phrase
        flatten_pred_relation_scores = flatten_pred_relation_scores[phrase_select]
        flatten_phrase_pred_subject_ids_list = flatten_phrase_pred_subject_ids_list[phrase_select]
        flatten_phrase_pred_object_ids_list = flatten_phrase_pred_object_ids_list[phrase_select]
        flatten_phrase_pred_subject_in_crowds_list = flatten_phrase_pred_subject_in_crowds_list[phrase_select]
        flatten_phrase_pred_object_in_crowds_list = flatten_phrase_pred_object_in_crowds_list[phrase_select]
        flatten_phrase_relation_ids_list = flatten_phrase_relation_ids_list[phrase_select]

        topk_scores, topk_flatten_indx = torch.sort(flatten_pred_relation_scores.flatten(), descending=True)
        subject_ids_list = []
        object_ids_list = []
        relation_ids = []
        scores = []
        triplet_ids_list=[]
        for i in range(topk_scores.shape[0]):
            if len(triplet_ids_list)>=100:
                break
            subject_ids = flatten_phrase_pred_subject_ids_list[topk_flatten_indx[i]]
            object_ids = flatten_phrase_pred_object_ids_list[topk_flatten_indx[i]]
            relation_id = flatten_phrase_relation_ids_list[topk_flatten_indx[i]].item()
            topk_score = topk_scores[i].item()

            subject_in_crowd_scores = flatten_phrase_pred_subject_in_crowds_list[topk_flatten_indx[i]]
            object_in_crowd_scores = flatten_phrase_pred_object_in_crowds_list[topk_flatten_indx[i]]
            # print(subject_in_crowd_scores)
            # print(object_in_crowd_scores)
            # print(subject_ids, subject_in_crowd_scores)
            # print(object_ids, object_in_crowd_scores)
            # print(topk_score)
            subject_idxs = torch.where(subject_in_crowd_scores > 0)[0]
            object_idxs = torch.where(object_in_crowd_scores > 0)[0]

            if subject_idxs.shape[0] > 0 and object_idxs.shape[0] > 0:
                single_subject_scores=subject_in_crowd_scores[subject_idxs].data.cpu().numpy().tolist()
                single_object_scores=object_in_crowd_scores[object_idxs].data.cpu().numpy().tolist()
                single_subject_ids=subject_ids[subject_idxs].data.cpu().numpy().tolist()
                single_object_ids=object_ids[object_idxs].data.cpu().numpy().tolist()
                single_subject_ids_only=list(set(single_subject_ids).difference(set(single_object_ids)))
                single_object_ids_only=list(set(single_object_ids).difference(set(single_subject_ids)))
                single_ids_intersection=set(single_subject_ids).intersection(set(single_object_ids))
                for idx in single_ids_intersection:
                    suj_score=single_subject_scores[single_subject_ids.index(idx)]
                    obj_score=single_object_scores[single_object_ids.index(idx)]
                    if suj_score >= obj_score:
                        single_subject_ids_only.append(idx)
                    elif suj_score < obj_score:
                        single_object_ids_only.append(idx)
                triplet_id=str(sorted(single_subject_ids_only))+"_"+str(sorted(single_object_ids_only))+"_"+str(relation_id)
                if len(single_subject_ids_only)==0 \
                        or len(single_object_ids_only)==0 \
                        or triplet_id in triplet_ids_list \
                        or (len(single_subject_ids_only)==1 and len(single_object_ids_only)==1):
                    continue
                replace=False
                if merge=="merge":
                    for exist_i,(exist_triplet_ids,exist_subject_ids,exist_object_ids,exist_relation_id,exist_relation_id) in enumerate(zip(triplet_ids_list,subject_ids_list,object_ids_list,relation_ids,relation_ids)):
                        if exist_relation_id==relation_id and \
                            len(set(single_subject_ids_only).difference(set(exist_subject_ids)))>=0 \
                            and len(set(exist_subject_ids).intersection(set(single_subject_ids_only)))>0 \
                            and len(set(single_object_ids_only).difference(set(exist_object_ids)))>=0 \
                            and len(set(exist_object_ids).intersection(set(single_object_ids_only)))>0:
                            single_subject_ids_only=list(set(single_subject_ids_only).union(set(exist_subject_ids)))
                            single_object_ids_only = list(set(single_object_ids_only).union(set(exist_object_ids)))
                            triplet_id=str(sorted(single_subject_ids_only))+"_"+str(sorted(single_object_ids_only))+"_"+str(relation_id)
                            triplet_ids_list[exist_i]=triplet_id
                            subject_ids_list[exist_i]=single_subject_ids_only
                            object_ids_list[exist_i]=single_object_ids_only
                            replace=True
                            break
                elif merge=="skip":
                    for exist_i,(exist_triplet_ids,exist_subject_ids,exist_object_ids,exist_relation_id,exist_relation_id) in enumerate(zip(triplet_ids_list,subject_ids_list,object_ids_list,relation_ids,relation_ids)):
                        if exist_relation_id==relation_id and \
                            len(set(single_subject_ids_only).difference(set(exist_subject_ids)))>=0 \
                            and len(set(exist_subject_ids).intersection(set(single_subject_ids_only)))>0 \
                            and len(set(single_object_ids_only).difference(set(exist_object_ids)))>=0 \
                            and len(set(exist_object_ids).intersection(set(single_object_ids_only)))>0:
                            # single_subject_ids_only=list(set(single_subject_ids_only).union(set(exist_subject_ids)))
                            # single_object_ids_only = list(set(single_object_ids_only).union(set(exist_object_ids)))
                            # triplet_id=str(sorted(single_subject_ids_only))+"_"+str(sorted(single_object_ids_only))+"_"+str(relation_id)
                            # triplet_ids_list[exist_i]=triplet_id
                            # subject_ids_list[exist_i]=single_subject_ids_only
                            # object_ids_list[exist_i]=single_object_ids_only
                            replace=True
                            break
                if not replace:
                    triplet_ids_list.append(triplet_id)
                    subject_ids_list.append(single_subject_ids_only)
                    object_ids_list.append(single_object_ids_only)
                    relation_ids.append(relation_id)
                    scores.append(topk_score)
                if len(triplet_ids_list) <= 5 and (len(subject_idxs) < torch.sum(subject_ids>=0).item() or len(object_idxs) < torch.sum(object_ids>=0).item()):
                    # print(image_id,i)
                    # print(subject_idxs)
                    # print(subject_ids)
                    # print(object_idxs)
                    # print(object_ids)
                    if visible:
                        image_info=image_dict[image_id]
                        img = read_image(image_path+"/" + image_id.zfill(12)+".jpg", format="RGB")

                        triplets_subject_classes = []
                        triplets_object_classes = []
                        triplets_subject_boxes = []
                        triplets_object_boxes = []
                        triplets_subject_masks = []
                        triplets_object_masks = []
                        for sub in single_subject_ids_only:
                            subject_instance = image_info['instances'][str(sub)]
                            triplets_subject_classes.append(subject_instance['instance_class_id'])
                            triplets_subject_boxes.append([subject_instance['box'][0], subject_instance['box'][1],
                                                           subject_instance['box'][2] + subject_instance['box'][0],
                                                           subject_instance['box'][3] + subject_instance['box'][1]])
                            triplets_subject_masks.append(subject_instance['segmentation'])
                        for obj in single_object_ids_only:
                            object_instance = image_info['instances'][str(obj)]
                            triplets_object_classes.append(object_instance['instance_class_id'])
                            triplets_object_boxes.append([object_instance['box'][0], object_instance['box'][1],
                                                          object_instance['box'][2] + object_instance['box'][0],
                                                          object_instance['box'][3] + object_instance['box'][1]])
                            triplets_object_masks.append(object_instance['segmentation'])

                        triplets_subject_classes = torch.Tensor(triplets_subject_classes).int() - 1
                        triplets_object_classes = torch.Tensor(triplets_object_classes).int() - 1
                        triplets_subject_boxes = Boxes(torch.Tensor(triplets_subject_boxes))
                        triplets_object_boxes = Boxes(torch.Tensor(triplets_object_boxes))
                        triplets_subject_masks = triplets_subject_masks
                        triplets_object_masks = triplets_object_masks
                        triplets_relations = torch.Tensor([relation_id]).int() - 1
                        triplets_scores = torch.Tensor([topk_score])

                        visualizer = Visualizer(img, MetadataCatalog.get(dataset_name), instance_mode=ColorMode.IMAGE)
                        vis_output_instance, relation = visualizer.draw_group_relation_predictions(image_info['height'],
                                                                                                   image_info['width'],
                                                                                                   triplets_subject_classes,
                                                                                                   triplets_object_classes,
                                                                                                   triplets_subject_boxes,
                                                                                                   triplets_object_boxes,
                                                                                                   triplets_subject_masks,
                                                                                                   triplets_object_masks,
                                                                                                   triplets_relations,
                                                                                                   triplets_scores)
                        vis_output_instance.save(os.path.join("vis",image_id+"_"+str(i)+".png"))
                        # exit()
                    if image_id not in valid_vis_images_ids:
                        valid_vis_images_ids.append(image_id)

        triplets_dict = {}
        # print(phrase_select[0].shape[0],len(scores))
        if len(scores) > 0:
            matched_gt=[]
            for i, (subject_ids, object_ids, relation_id, score) in enumerate(
                    zip(subject_ids_list, object_ids_list, relation_ids, scores)):
                triplets_dict[str(i)] = {
                    "triplet_id": i,
                    "subject_instance": subject_ids,
                    "object_instance": object_ids,
                    "relation_id": relation_id + 1,
                    "score": score
                }
                subject_cls=pred_image_instances[str(subject_ids[0])]['instance_class_id']
                object_cls = pred_image_instances[str(object_ids[0])]['instance_class_id']
                matched=False
                for k in gt_triplets:
                    if k in matched_gt:
                        continue
                    if (pred_image_instances[str(subject_ids[0])]['instance_class_id'], relation_id+1, pred_image_instances[str(object_ids[0])]['instance_class_id']) \
                            != (gt_image_instances[str(gt_triplets[k]['subject_instance'][0])]['instance_class_id'], gt_triplets[k]['relation_id'], gt_image_instances[str(gt_triplets[k]['object_instance'][0])]['instance_class_id']):
                        continue

                    gt_subj_crowd_mask = union_boxes_to_mask([gt_image_instances[str(gt_ins_id)]['box'] for gt_ins_id in gt_triplets[k]['subject_instance']], gt_img_h, gt_img_w)
                    gt_obj_crowd_mask = union_boxes_to_mask([gt_image_instances[str(gt_ins_id)]['box'] for gt_ins_id in gt_triplets[k]['object_instance']], gt_img_h, gt_img_w)
                    pred_subj_crowd_mask = union_boxes_to_mask([pred_image_instances[str(pred_ins_id)]['box'] for pred_ins_id in subject_ids], gt_img_h, gt_img_w)
                    pred_obj_crowd_mask = union_boxes_to_mask([pred_image_instances[str(pred_ins_id)]['box'] for pred_ins_id in object_ids], gt_img_h, gt_img_w)

                    pred_subj_crowd_mask = cv2.resize(pred_subj_crowd_mask, (gt_img_w, gt_img_h), cv2.INTER_NEAREST)
                    pred_obj_crowd_mask = cv2.resize(pred_obj_crowd_mask, (gt_img_w, gt_img_h), cv2.INTER_NEAREST)

                    subj_iou = compute_pixel_iou(pred_subj_crowd_mask, gt_subj_crowd_mask)
                    obj_iou = compute_pixel_iou(pred_obj_crowd_mask, gt_obj_crowd_mask)
                    # 这里的匹配是iou超过阈值就可以了，并没有找最匹配的，因为可能subj_mask是subj_iou_max，但是obj_mask不是, 而max的subj和obj之间又没有关系
                    if subj_iou > 0.5 and obj_iou > 0.5:
                        matched_gt.append(k)
                        matched=True
                        image_count[image_id].append([subject_ids,object_ids,relation_id+1,score])
                        if len(subject_ids) < cls_dict[subject_cls] or len(object_ids) < cls_dict[object_cls]:
                            if image_id not in matched_valid_vis_images_ids:
                                matched_valid_vis_images_ids.append(image_id)

        image_triplet_dict[image_id]['triplets']=triplets_dict
        processbar.update(1)
    processbar.close()
    result_f.close()
    if len(merge)>0:
        json.dump(image_triplet_dict, open(os.path.join(cfg.OUTPUT_DIR, "test_images_triplets_dict"+str(in_crowd_threshold)+less+"_"+merge+".json"), 'w'))
        json.dump(valid_vis_images_ids,open(os.path.join(cfg.OUTPUT_DIR, "valid_vis_image_ids_"+str(in_crowd_threshold)+less+"_"+merge+".json"), 'w'))
    else:
        json.dump(image_triplet_dict, open(os.path.join(cfg.OUTPUT_DIR, "test_images_triplets_dict"+str(in_crowd_threshold)+less+".json"), 'w'))
        json.dump(valid_vis_images_ids,open(os.path.join(cfg.OUTPUT_DIR, "valid_vis_image_ids_"+str(in_crowd_threshold)+less+".json"), 'w'))
    json.dump(image_count,open(os.path.join(cfg.OUTPUT_DIR, "test"+str(in_crowd_threshold)+".json"),'w'))
    json.dump(matched_valid_vis_images_ids,open(os.path.join(cfg.OUTPUT_DIR, "valid100.json"),'w'))

def convert_box_to_mask(box, img_h, img_w):
    # box:x,y,w,h
    mask = np.zeros([img_h, img_w])
    #print(box)
    x = int(box[0])
    y = int(box[1])
    w = int(box[2])
    h = int(box[3])
    #print(img_h,img_w)
    #print(box)
    #for i in range(y, y + h):
    #    for j in range(x, x + w):
    #        if mask[i][j] == 0:
    #            mask[i][j] = 1
    mask[y:y+h,x:x+w] = 1
    return mask

def union_boxes_to_mask(boxes, img_h, img_w):
    sum_mask = np.zeros([img_h, img_w])
    for box in boxes:
        #box:x,y,w,h
        #print(box)
        mask = convert_box_to_mask(box, img_h, img_w)
        sum_mask = sum_mask + mask
    return sum_mask

def compute_pixel_iou(mask_pred, mask_gt):
    intersection=mask_pred*mask_gt
    union=mask_pred+mask_gt
    return np.count_nonzero(intersection)/np.count_nonzero(union)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def main(args):
    start = time.time()
    cfg = setup(args)

    model = build_model(cfg)
    model.eval()

    if args.mode=="train_crowd":
        do_crowd_train(cfg,model,args.resume,args.print_size)
    elif args.mode=="train_phrase":
        do_phrase_train(cfg,model,args.resume,args.print_size)
    elif args.mode=="train_relation":
        do_relation_train(cfg, model,args.resume,args.print_size)
    elif args.mode=="test_relation":
        do_relation_test(cfg,model,args.visible,args.print_size)
        generate_triplets(cfg,args.crth,args.visible,args.merge)
    elif args.mode=="generate_triplets":
        generate_triplets(cfg,args.crth,args.visible,args.merge)
    else:
        print("mode not supported")
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    # print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
