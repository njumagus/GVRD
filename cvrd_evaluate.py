#如果按instance分割结果评估的话，匹配更严格，就是计算群体中的每个instance的iou了,个数必须全对，然后匹配过程中先把最大iou的拿出来，然后再继续

import json
import argparse
import numpy as np
import cv2
import time
from collections import defaultdict
from pycocotools import mask as maskUtils
import os
import csv
import copy
from tqdm import tqdm

class MatrixTri():
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, matrix_tri):
        self.tp += matrix_tri.tp
        self.fp += matrix_tri.fp
        self.fn += matrix_tri.fn
        return self

    def precision(self, mAP=False):
        if mAP and self.tp + self.fn == 0:
            return -1
        if self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def pred_num(self,):
        return self.tp+self.fp

    def gt_num(self,):
        return self.tp+self.fn

class Matrix():
    def __init__(self):
        self.matrix_per_tri = defaultdict(MatrixTri)

    def __getitem__(self, tri):
        return self.matrix_per_tri[tri]

    def __iadd__(self, matrix):
        for tri, matrix_tri in matrix.matrix_per_tri.items():
            self.matrix_per_tri[tri] += matrix_tri
        return self

    def metrics(self, beta2, predicate_count,predicate_pred_counts,predicate_gt_counts,predicate_num):
        matrix_all = MatrixTri()

        for matrix_tri in self.matrix_per_tri.values():
            matrix_all += matrix_tri

        precision = matrix_all.precision()
        recall = matrix_all.recall()

        if precision == 0 and recall == 0:
            f = 0
        else:
            f = (beta2 + 1) * precision * recall / (beta2 * precision + recall)

        pred_num = matrix_all.pred_num()
        gt_num = matrix_all.gt_num()
        mean_precisions,mean_recalls=[],[]
        mean_precision_sum,mean_recall_sum=0,0
        for key in predicate_count:
            p_precision=0
            if predicate_count[key]>0:
                p_precision=predicate_count[key] / predicate_pred_counts[key]
            mean_precision_sum +=p_precision
            p_recall=0
            if predicate_count[key]>0:
                p_recall=predicate_count[key] / predicate_gt_counts[key]
            mean_recall_sum+=p_recall
            mean_precisions.append([key,p_precision * 100])
            mean_recalls.append([key, p_recall * 100])

        return {
            'precision': precision,
            'recall': recall,
            'p-precisions': mean_precisions,
            'p-recalls': mean_recalls,
            'm-precision': mean_precision_sum/predicate_num,
            'm-recall': mean_recall_sum/predicate_num,
            'f': f,
        }


def merge_image_triplet_info(image_info, triplet_info, is_pred = False):
    instances = image_info['instances']
    triplets = triplet_info['triplets']
    relas = []
    subjs = []
    subjs_segs = []
    subjs_boxes = []
    objs = []
    objs_segs = []
    objs_boxes = []
    scores = []
    for triplet_id in triplets:
        triplet = triplets[triplet_id]
        relas.append(triplet['relation_id'])
        if is_pred:
            scores.append(triplet['score'])
        subj = -1
        subj_segs = []
        subj_boxes = []
        obj = -1
        obj_segs = []
        obj_boxes = []
        if 'subject_instance' in triplet:
            triplet_subject_instances=triplet['subject_instance']
        elif 'subject_instance_id' in triplet:
            triplet_subject_instances = triplet['subject_instance_id']
        for instance_id in triplet_subject_instances:
            tmp_instance = instances[str(instance_id)]
            subj = tmp_instance['instance_class_id']
            # subj_seg = tmp_instance['segmentation']
            subj_box = tmp_instance['box']
            # if is_pred:
            #     if subj_seg['counts'][0:2] == "b'" and subj_seg['counts'][len(subj_seg['counts'])-1]=="'":
            #         pred的文件中的segmentation的格式有点问题，先转换一下
                    # converted_seg_counts = subj_seg['counts'][2:len(subj_seg['counts']) - 1]
                    # converted_seg_counts = eval(repr(converted_seg_counts).replace('\\\\', '\\'))
                    # converted_subj_seg = {'counts':converted_seg_counts, 'size': subj_seg['size']}
                    # subj_segs.append(converted_subj_seg)
                    # continue
            #
            # subj_segs.append(subj_seg)
            subj_boxes.append(subj_box)
        if 'object_instance' in triplet:
            triplet_object_instances=triplet['object_instance']
        elif 'object_instance_id' in triplet:
            triplet_object_instances = triplet['object_instance_id']
        for instance_id in triplet_object_instances:
            tmp_instance = instances[str(instance_id)]
            obj = tmp_instance['instance_class_id']
            # obj_seg = tmp_instance['segmentation']
            obj_box = tmp_instance['box']
            # if is_pred:
            #     if obj_seg['counts'][0:2] == "b'" and obj_seg['counts'][len(obj_seg['counts']) - 1] == "'":
            #         converted_seg_counts = obj_seg['counts'][2:len(obj_seg['counts'])-1]
            #         converted_seg_counts = eval(repr(converted_seg_counts).replace('\\\\', '\\'))
            #         converted_obj_seg = {'counts':converted_seg_counts, 'size':obj_seg['size']}
            #         obj_segs.append(converted_obj_seg)
            #         continue
            # obj_segs.append(obj_seg)
            #print(obj_box)
            obj_boxes.append(obj_box)
        subjs.append(subj)
        # subjs_segs.append(subj_segs)
        subjs_boxes.append(subj_boxes)
        objs.append(obj)
        objs_segs.append(obj_segs)
        objs_boxes.append(obj_boxes)
        #print(obj_boxes)
    if is_pred:
        return relas, subjs, subjs_segs, subjs_boxes, objs, objs_segs, objs_boxes, scores
    else:
        return relas, subjs, subjs_segs, subjs_boxes, objs, objs_segs, objs_boxes

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

def compute_crowd_matrix(gt_image, gt_triplet, pred_image, pred_triplet, _top_ns, iou_thresh, predicate_counts, predicate_gt_counts, predicate_pred_counts):

    gt_relas, gt_subjs, gt_subjs_segs, gt_subjs_boxes, gt_objs, gt_objs_segs, gt_objs_boxes = merge_image_triplet_info(gt_image, gt_triplet)
    pred_relas, pred_subjs, pred_subjs_segs, pred_subjs_boxes, pred_objs, pred_objs_segs, pred_objs_boxes, scores = merge_image_triplet_info(pred_image, pred_triplet, is_pred=True)
    gt_img_h = gt_image['height']
    gt_img_w = gt_image['width']
    pred_img_h = pred_image['height']
    pred_img_w = pred_image['width']
    num_gt = len(gt_relas)
    num_pred = len(pred_relas)

    top_ns = copy.deepcopy(_top_ns)

    indices = sorted(range(num_pred), key=lambda i: -scores[i])[:max(top_ns)]

    matrixs, max_tps, matched_gts, matched_preds = [],[],[],[]
    for i in top_ns:
        matrixs.append(Matrix())
        max_tps.append(-1)
        matched_gts.append(set())
        matched_preds.append(set())

    matrix = Matrix()
    matched_gt = set()
    matched_pred = set()
    matched_predicate = defaultdict(int)
    all_predicate_pred_counts = defaultdict(int)
    for gt_r in gt_relas:
        predicate_gt_counts[gt_r]+=1

    for index,j in enumerate(indices):
        all_predicate_pred_counts[pred_relas[j]] += 1
        for k in range(num_gt):
            if k in matched_gt:
                continue
            if (pred_subjs[j], pred_relas[j], pred_objs[j]) != (gt_subjs[k], gt_relas[k], gt_objs[k]):
                continue

            gt_subj_crowd_mask = union_boxes_to_mask(gt_subjs_boxes[k], gt_img_h, gt_img_w)
            
            gt_obj_crowd_mask = union_boxes_to_mask(gt_objs_boxes[k], gt_img_h, gt_img_w)
            pred_subj_crowd_mask = union_boxes_to_mask(pred_subjs_boxes[j], pred_img_h, pred_img_w)
            pred_obj_crowd_mask = union_boxes_to_mask(pred_objs_boxes[j], pred_img_h, pred_img_w)

            pred_subj_crowd_mask = cv2.resize(pred_subj_crowd_mask, (gt_img_w,gt_img_h),cv2.INTER_NEAREST)
            pred_obj_crowd_mask = cv2.resize(pred_obj_crowd_mask, (gt_img_w,gt_img_h),cv2.INTER_NEAREST)
            
            subj_iou = compute_pixel_iou(pred_subj_crowd_mask, gt_subj_crowd_mask)
            obj_iou = compute_pixel_iou(pred_obj_crowd_mask, gt_obj_crowd_mask)
            if subj_iou > iou_thresh and obj_iou > iou_thresh:
                matched_gt.add(k)
                matched_pred.add(j)
                matrix[(gt_subjs[k], gt_relas[k], gt_objs[k])].tp += 1
                matched_predicate[gt_relas[k]]+=1
                # visualization(gt_image, j, pred_subjs[j], pred_subjs_boxes[j], pred_objs[j], pred_objs_boxes[j] , pred_relas[j])
                break

        if (index + 1) in top_ns:
            top_n_index = top_ns.index(index + 1)
            max_tps[top_n_index] = min(index + 1, num_gt)
            matrixs[top_n_index] = copy.deepcopy(matrix)
            matched_gts[top_n_index] = matched_gt.copy()
            matched_preds[top_n_index] = matched_pred.copy()
            for key in matched_predicate:
                predicate_counts[top_n_index][key]+=matched_predicate[key]
            for key in all_predicate_pred_counts:
                predicate_pred_counts[top_n_index][key]+=all_predicate_pred_counts[key]

    for i, top_n in enumerate(top_ns):
        if max_tps[i] < 0:
            max_tps[i] = min(top_n, num_gt)
            matrixs[i] = copy.deepcopy(matrix)
            matched_gts[i] = matched_gt.copy()
            matched_preds[i] = matched_pred.copy()

    for i, top_n in enumerate(top_ns):
        if top_n == -1:
            continue
        for j in indices[:top_n]:
            if j not in matched_preds[i]:
                matrixs[i][(pred_subjs[j], pred_relas[j], pred_objs[j])].fp += 1

        for k in range(num_gt):
            if k not in matched_gts[i]:
                matrixs[i][(gt_subjs[k], gt_relas[k], gt_objs[k])].fn += 1

    return matrixs


def precompute_metrics(gt_image_dict, gt_triplet_dict, pred_image_dict, pred_triplet_dict, top_ns, crowd_iou_thresh, relation_dict):
    crowd_matrixs, crowd_predicate_counts = [], []
    crowd_predicate_gt_counts, crowd_predicate_pred_counts = {int(class_id):0 for class_id in relation_dict}, []
    for i in top_ns:
        crowd_matrixs.append(Matrix())
        crowd_predicate_counts.append({int(class_id):0 for class_id in relation_dict})
        crowd_predicate_pred_counts.append({int(class_id):0 for class_id in relation_dict})

    process_bar=tqdm(total=len(gt_image_dict),unit="image")
    for img_id in gt_image_dict:
        if (img_id not in pred_image_dict) or (img_id not in pred_triplet_dict):
            print('img not in pred')
            exit()
        else:
            tmp_crowd_matrixs = compute_crowd_matrix(gt_image_dict[img_id], gt_triplet_dict[img_id], pred_image_dict[img_id], pred_triplet_dict[img_id], top_ns, crowd_iou_thresh, crowd_predicate_counts,crowd_predicate_gt_counts,crowd_predicate_pred_counts)

        for i in range(0,len(top_ns)):
            crowd_matrixs[i] += tmp_crowd_matrixs[i]

        process_bar.update(1)
    process_bar.close()
    return crowd_matrixs,crowd_predicate_counts,crowd_predicate_pred_counts,crowd_predicate_gt_counts

def compute_metrics(beta2,relation_dict,
                    crowd_matrixs,crowd_predicate_counts,crowd_predicate_pred_counts,crowd_predicate_gt_counts):
    predicate_num=len(relation_dict)
    crowd_reuslts = []
    for i in range(0,len(top_ns)):
        crowd_reuslts.append(crowd_matrixs[i].metrics(beta2, crowd_predicate_counts[i], crowd_predicate_pred_counts[i], crowd_predicate_gt_counts,predicate_num))

    return crowd_reuslts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate precision, recall and f.')
    parser.add_argument('--func_name', dest='func_name', type=str,
                        default='statistics_baseline_anchor',
                        help='gt json file path, default: "data/test/images_dict.json"')
    parser.add_argument('--gt_image_dict_json', dest='gt_image_dict_json', type=str, default='../data/gvrd/test/images_dict.json',
                        help='gt json file path, default: "../data/gvrd/test/images_dict.json"')
    parser.add_argument('--gt_triplet_dict_json', dest='gt_triplet_dict_json', type=str, default='../data/gvrd/test/images_triplets_dict.json',
                        help='gt json file path, default: "../data/gvrd/test/images_triplets_dict.json"')
    parser.add_argument('--config', dest='config', type=str, default=None,
                        help='configuration, default: "relation_clust_instance_anchor_attention"')
    parser.add_argument('--pred_image_dict_json', dest='pred_image_dict_json', type=str, default='output/test_images_dict.json',
                        help='gt json file path, default: "output/test_images_dict.json"')
    parser.add_argument('--pred_triplet_dict_json', dest='pred_triplet_dict_json', type=str, default='output/test_images_triplets_dict.json',
                        help='gt json file path, default: "output/test_images_triplets_dict.json"')
    # parser.add_argument('--top_n', dest='top_n', type=int, default=0,
    #                     help='precision/recall @ top-n scored relations, 0 means top-α, default: 0')
    parser.add_argument('--beta2', dest='beta2', type=float, choices=[0.3, 1], default=0.3,
                        help='beta2 of F-score, 0.3 or 1, default: 0.3')
    parser.add_argument('--crowd_iou_thresh', dest='crowd_iou_thresh', type=float, default=0.5,
                        help='crowd bounding box IOU threshold, default: 0.5')
    parser.add_argument('--crth', dest='crth', type=float, default=0.5,
                        help='instance bounding box IOU threshold, default: 0.5')
    parser.add_argument("--noless", action="store_true", help="Whether to show result")
    parser.add_argument("--merge", type=str,default="")
    parser.add_argument("--baseline", action="store_true", default=False,help="Whether to use baseine")
    args = parser.parse_args()

    top_ns = [10,20,30]
    #top_n = args.top_n if args.top_n != 0 else 'α'
    print('-- Config --')
    print('func name:', args.func_name)
    print('gt image dict json:', args.gt_image_dict_json)
    print('gt triplet dict json:', args.gt_triplet_dict_json)
    print('pred image dict json:', args.pred_image_dict_json)
    print('pred triplet dict json:', args.pred_triplet_dict_json)
    print('precision/recall @ top-n:', str(top_ns))
    print('F-score beta2:', args.beta2)
    print('Crowd IOU threshold:', args.crowd_iou_thresh)
    print()

    result = [str(args.crth)+"_"+args.func_name]
    if len(args.merge)>0:
        result[0]=result[0]+"_"+args.merge
    less="_less"
    if args.noless:
        less=""

    gt_image_dict = json.load(open(args.gt_image_dict_json))
    gt_triplet_dict = json.load(open(args.gt_triplet_dict_json))
    if args.func_name and not args.baseline:
        pred_image_dict = json.load(open(args.func_name + "/test_images_dict" + less + ".json"))
        if len(args.merge)>0:
            pred_triplet_dict = json.load(open(args.func_name + "/test_images_triplets_dict" + str(args.crth) + less + "_"+args.merge+".json"))
        else:
            pred_triplet_dict = json.load(open(args.func_name+"/test_images_triplets_dict"+str(args.crth)+less+".json"))
    elif args.func_name and args.baseline:
        pred_image_dict = json.load(open(args.func_name + "/test_images_dict.json")) #"baselines/" +
        if len(args.merge)>0:
            pred_triplet_dict = json.load(open(args.func_name + "/test_images_triplets_dict_"+args.merge+".json")) #"baselines/" +
        else:
            pred_triplet_dict = json.load(open(args.func_name + "/test_images_triplets_dict.json")) #"baselines/"+
    else:
        pred_image_dict = json.load(open(args.pred_image_dict_json))
        pred_triplet_dict = json.load(open(args.pred_triplet_dict_json))

    max_triplet_num = 0
    min_triplet_num = 100000
    less_than_100 = 0
    for img_id in pred_triplet_dict:
        triplets_num = len(list(pred_triplet_dict[img_id]['triplets']))
        if triplets_num > max_triplet_num:
            max_triplet_num = triplets_num
        if triplets_num < min_triplet_num:
            min_triplet_num = triplets_num
        if triplets_num < 100:
            less_than_100 += 1

    print('max crowd triplet num:' + str(max_triplet_num))
    print('min crowd triplet num:' + str(min_triplet_num))
    print('triplet num less than 100:' + str(less_than_100))

    relation_dict=json.load(open("../data/gvrd/relation_dict.json",'r'))

    for image_id in gt_triplet_dict:
        if image_id not in pred_triplet_dict:
            pred_triplet_dict[image_id]={"triplets":{}}
        if image_id not in pred_image_dict:
            pred_image_dict[image_id]={"instances":{},"height":gt_image_dict[image_id]['height'],"width":gt_image_dict[image_id]["width"]}

    result_csv_headers = ['func_name']
    crowd_matrixs, crowd_predicate_counts, crowd_predicate_pred_counts, crowd_predicate_gt_counts \
    = precompute_metrics(gt_image_dict, gt_triplet_dict, pred_image_dict,pred_triplet_dict,
                        top_ns, args.crowd_iou_thresh,relation_dict)


    crowd_metrics=compute_metrics(args.beta2,relation_dict,
        copy.deepcopy(crowd_matrixs), copy.deepcopy(crowd_predicate_counts), copy.deepcopy(crowd_predicate_pred_counts), copy.deepcopy(crowd_predicate_gt_counts))
    for i,top_n in enumerate(top_ns):
        result_csv_headers.append('R-g@'+str(top_n))
        result_csv_headers.append('mR-g@' + str(top_n))

        crowd_metric = crowd_metrics[i]
        print('R@{}: {:.2f}'.format(top_n, crowd_metric['recall'] * 100))
        print('m-R@{}: {:.2f}'.format(top_n, crowd_metric['m-recall'] * 100))
        result.append(round(crowd_metric['recall'] * 100,2))
        result.append(round(crowd_metric['m-recall'] * 100,2))

    evaluate_file_path = 'cvrd_evaluate_result'+less+'_box_detail.csv'
    if os.path.exists(evaluate_file_path):
        result_file = csv.writer(open(evaluate_file_path, 'a+'))
    else:
        result_file = csv.writer(open(evaluate_file_path,'w'))
        result_file.writerow(result_csv_headers)
    result_file.writerows([result])


