# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import json
import os
import datetime
import numpy as np
from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager, file_lock
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
"""
This file contains functions to register a COCO-format dataset to the DatasetCatalog.
"""

__all__ = ["register_cvrd","load_gvrd_json"]

logger = logging.getLogger(__name__)

def register_cvrd(name, metadata,
                image_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_gvrd_json(name,
                image_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        image_path=image_path,
        class_json_file=class_json_file,
        relation_json_file=relation_json_file,
        instance_json_file=instance_json_file,
        triplet_json_file=triplet_json_file, evaluator_type="cvrd", **metadata
    )

def register_cvrd_duplicated(name, metadata,
                image_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_cvrd_duplicated_json(name,
                image_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        image_path=image_path,
        class_json_file=class_json_file,
        relation_json_file=relation_json_file,
        instance_json_file=instance_json_file,
        triplet_json_file=triplet_json_file, evaluator_type="cvrd", **metadata
    )

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class CVRD:
    def __init__(self,image_path,class_json_file,relation_json_file,instance_json_file,triplet_json_file):
        self.image_path=image_path
        self.class_dict=json.load(open(class_json_file))
        self.relation_dict=json.load(open(relation_json_file))
        self.image_instance_dict=json.load(open(instance_json_file))
        self.image_triplet_dict = json.load(open(triplet_json_file))

        self.thing_list = []
        for class_id in range(1,len(self.class_dict)+1):
            self.thing_list.append(self.class_dict[str(class_id)]) # list of thing dicts from 1 to 80

        self.image_id_list=sorted([int(image_id) for image_id in list(self.image_instance_dict.keys())])

    def loadClassdict(self):
        return self.class_dict

    def loadThings(self):
        return self.thing_list

    def loadIds(self):
        return self.image_id_list

    def loadImgs(self,ids):
        if _isArrayLike(ids):
            return [self.image_instance_dict[str(id)] for id in ids]
        elif type(ids) == int:
            return [self.image_instance_dict[str(ids)]]

    def loadInstances(self,ids):
        if _isArrayLike(ids):
            return [self.image_instance_dict[str(id)]['instances'] for id in ids]
        elif type(ids) == int:
            return [self.image_instance_dict[str(ids)]['instances']]

    def loadTriplets(self,ids):
        if _isArrayLike(ids):
            return [self.image_triplet_dict[str(id)]['triplets'] for id in ids]
        elif type(ids) == int:
            return [self.image_triplet_dict[str(ids)]['triplets']]


def load_gvrd_json(dataset_name,
                image_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file,
                extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation annotations.

    Args:
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    timer = Timer()
    api=CVRD(image_path,class_json_file,relation_json_file,instance_json_file,triplet_json_file)
    # if timer.seconds() > 1:
    #     logger.info("Loading cvrd takes {:.2f} seconds.".format(timer.seconds()))


    meta = MetadataCatalog.get(dataset_name)
    thing_dataset_id_to_contiguous_id=meta.get("thing_dataset_id_to_contiguous_id")
    relation_dataset_id_to_contiguous_id=meta.get("relation_dataset_id_to_contiguous_id")
    relation_classes=meta.get("relation_classes")
    # logger.info("Loaded {} images in CVRD".format(len(api.image_instance_dict)))

    # ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])
    dataset_dict = []
    image_ids=api.loadIds()
    for image_id in image_ids:
        img_dict = api.loadImgs(image_id)[0]
        record = {}
        record["file_name"] = os.path.join(image_path, img_dict["image_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["image_id"]

        instance_dict=img_dict['instances']
        objs = []
        object_id_list=[]
        thing_count=0
        # interest_map=np.zeros((img_dict["height"],img_dict["width"]))
        # print("=================")
        # print(image_id)
        # print([instance_dict[instance_id]['instance_class_id'] for instance_id in instance_dict])
        for instance_id in instance_dict:
            instance=instance_dict[instance_id]

            object_id_list.append(instance_id)
            obj = {}
            obj['iscrowd']=instance['iscrowd']
            obj['bbox']=instance['box']
            obj['category_id']=thing_dataset_id_to_contiguous_id[api.loadClassdict()[str(instance['instance_class_id'])]['category_id']]
            # obj['category_name'] = api.loadClassdict()[str(instance['instance_class_id'])]['name']
            # obj['class_id'] = instance['instance_class_id']
            obj['segmentation']=instance['segmentation']
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            # if obj['labeled']==1:
            #     mask=mask_utils.decode(instance['segmentation'])
            #     interest_map[mask==1]=255
            objs.append(obj)
            thing_count+=1
        record["annotations"] = objs
        # record['interest_map'] = interest_map
        # Image.fromarray(interest_map).convert('L').save("interest_map.png")
        record["instance_ids"] = object_id_list
        instance_ids=[]
        for id in object_id_list:
            instance_ids.append(id)
        # print(instance_ids)
        triplets = api.loadTriplets(image_id)[0]

        triplet_records=[]
        # print([triplets[triplet_id]['subject_instance'] for triplet_id in triplets])
        # print([triplets[triplet_id]['object_instance'] for triplet_id in triplets])
        maxlen=sorted([max(len(triplets[triplet_id]['subject_instance']),len(triplets[triplet_id]['object_instance'])) for triplet_id in triplets])[-1]

        relation_onehot_dict={}
        sub_obj_dict={}
        triplet_ids=[]
        for triplet_id in triplets:
            triplet_ids.append(triplet_id)
            triplet=triplets[triplet_id]
            sub_ids=[]
            obj_ids=[]
            for l in range(maxlen):
                if l<len(triplet['subject_instance']):
                    sub_id=triplet['subject_instance'][l]
                    sub_ids.append(instance_ids.index(str(sub_id)))
                else:
                    sub_ids.append(-1)
            for l in range(maxlen):
                if l < len(triplet['object_instance']):
                    obj_id=triplet['object_instance'][l]
                    obj_ids.append(instance_ids.index(str(obj_id)))
                else:
                    obj_ids.append(-1)

            sub_obj_ids_str=""
            for sub_id_ in sub_ids:
                sub_obj_ids_str+=str(sub_id_).zfill(2)
            sub_obj_ids_str+="_"
            for obj_id_ in obj_ids:
                sub_obj_ids_str+=str(obj_id_).zfill(2)
            if sub_obj_ids_str not in sub_obj_dict:
                sub_obj_dict[sub_obj_ids_str]=[sub_ids,obj_ids]
            if sub_obj_ids_str not in relation_onehot_dict:
                relation_onehot_dict[sub_obj_ids_str]=np.zeros(len(relation_classes)+1)
            relation_onehot_dict[sub_obj_ids_str][relation_dataset_id_to_contiguous_id[triplet['relation_id']]]=1

        for sub_obj_ids_str in sub_obj_dict:
            tri={}

            tri['subject_ids']=sub_obj_dict[sub_obj_ids_str][0]
            tri['object_ids']=sub_obj_dict[sub_obj_ids_str][1]
            tri['relation_onehot']=relation_onehot_dict[sub_obj_ids_str].tolist()
            triplet_records.append(tri)

        record["triplets"] = triplet_records
        dataset_dict.append(record)
    return dataset_dict


def load_cvrd_duplicated_json(dataset_name,
                   image_path,
                   class_json_file,
                   relation_json_file,
                   instance_json_file,
                   triplet_json_file,
                   extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation annotations.

    Args:
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    timer = Timer()
    api = CVRD(image_path, class_json_file, relation_json_file, instance_json_file, triplet_json_file)
    # if timer.seconds() > 1:
    #     logger.info("Loading cvrd takes {:.2f} seconds.".format(timer.seconds()))

    meta = MetadataCatalog.get(dataset_name)
    thing_dataset_id_to_contiguous_id = meta.get("thing_dataset_id_to_contiguous_id")
    relation_dataset_id_to_contiguous_id = meta.get("relation_dataset_id_to_contiguous_id")
    relation_classes = meta.get("relation_classes")
    # logger.info("Loaded {} images in CVRD".format(len(api.image_instance_dict)))

    # ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])
    dataset_dict = []
    image_ids = api.loadIds()
    for image_id in image_ids:
        img_dict = api.loadImgs(image_id)[0]
        record = {}
        record["file_name"] = os.path.join(image_path, img_dict["image_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["image_id"]

        instance_dict = img_dict['instances']
        objs = []
        object_id_list = []
        thing_count = 0
        # interest_map=np.zeros((img_dict["height"],img_dict["width"]))
        # print("=================")
        # print(image_id)
        # print([instance_dict[instance_id]['instance_class_id'] for instance_id in instance_dict])
        for instance_id in instance_dict:
            instance = instance_dict[instance_id]

            object_id_list.append(instance_id)
            obj = {}
            obj['iscrowd'] = instance['iscrowd']
            obj['bbox'] = instance['box']
            obj['category_id'] = thing_dataset_id_to_contiguous_id[
                api.loadClassdict()[str(instance['instance_class_id'])]['category_id']]
            # obj['category_name'] = api.loadClassdict()[str(instance['instance_class_id'])]['name']
            # obj['class_id'] = instance['instance_class_id']
            obj['segmentation'] = instance['segmentation']
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            # if obj['labeled']==1:
            #     mask=mask_utils.decode(instance['segmentation'])
            #     interest_map[mask==1]=255
            objs.append(obj)
            thing_count += 1
        record["annotations"] = objs
        # record['interest_map'] = interest_map
        # Image.fromarray(interest_map).convert('L').save("interest_map.png")
        record["instance_ids"] = object_id_list
        instance_ids = []
        for id in object_id_list:
            instance_ids.append(id)
        # print(instance_ids)
        triplets = api.loadTriplets(image_id)[0]

        crowds = []
        for trip_id in triplets:
            if triplets[trip_id]['subject_instance'] not in crowds:
                # if len(triplets[trip_id]['subject_instance'])>1:
                crowds.append(triplets[trip_id]['subject_instance'])
            if triplets[trip_id]['object_instance'] not in crowds:
                # if len(triplets[trip_id]['object_instance']) > 1:
                crowds.append(triplets[trip_id]['object_instance'])
        duplicated = False
        for before_i, crowd_before in enumerate(crowds):
            for after_i, crowd_after in enumerate(crowds):
                if before_i != after_i:
                    intersec = set(crowd_before).intersection(set(crowd_after))
                    if len(intersec)>0 and (len(intersec) < len(crowd_before) or len(intersec) < len(crowd_after)):
                        duplicated = True
                        break
            if duplicated:
                break
        if duplicated:
            triplet_records = []
            # print([triplets[triplet_id]['subject_instance'] for triplet_id in triplets])
            # print([triplets[triplet_id]['object_instance'] for triplet_id in triplets])
            maxlen = sorted(
                [max(len(triplets[triplet_id]['subject_instance']), len(triplets[triplet_id]['object_instance'])) for
                 triplet_id in triplets])[-1]

            relation_onehot_dict = {}
            sub_obj_dict = {}
            triplet_ids = []
            for triplet_id in triplets:
                triplet_ids.append(triplet_id)
                triplet = triplets[triplet_id]
                sub_ids = []
                obj_ids = []
                for l in range(maxlen):
                    if l < len(triplet['subject_instance']):
                        sub_id = triplet['subject_instance'][l]
                        sub_ids.append(instance_ids.index(str(sub_id)))
                    else:
                        sub_ids.append(-1)
                for l in range(maxlen):
                    if l < len(triplet['object_instance']):
                        obj_id = triplet['object_instance'][l]
                        obj_ids.append(instance_ids.index(str(obj_id)))
                    else:
                        obj_ids.append(-1)

                sub_obj_ids_str = ""
                for sub_id_ in sub_ids:
                    sub_obj_ids_str += str(sub_id_).zfill(2)
                sub_obj_ids_str += "_"
                for obj_id_ in obj_ids:
                    sub_obj_ids_str += str(obj_id_).zfill(2)
                if sub_obj_ids_str not in sub_obj_dict:
                    sub_obj_dict[sub_obj_ids_str] = [sub_ids, obj_ids]
                if sub_obj_ids_str not in relation_onehot_dict:
                    relation_onehot_dict[sub_obj_ids_str] = np.zeros(len(relation_classes) + 1)
                relation_onehot_dict[sub_obj_ids_str][relation_dataset_id_to_contiguous_id[triplet['relation_id']]] = 1

            for sub_obj_ids_str in sub_obj_dict:
                tri = {}

                tri['subject_ids'] = sub_obj_dict[sub_obj_ids_str][0]
                tri['object_ids'] = sub_obj_dict[sub_obj_ids_str][1]
                tri['relation_onehot'] = relation_onehot_dict[sub_obj_ids_str].tolist()
                triplet_records.append(tri)

            record["triplets"] = triplet_records
            dataset_dict.append(record)
    return dataset_dict