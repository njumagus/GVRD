# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from .cvrd import register_cvrd, register_cvrd_duplicated
from .builtin_meta import _get_builtin_metadata


# ==== Predefined datasets and splits for COCO ==========
_PREDEFINED_SPLITS_CVRD = {}
_PREDEFINED_SPLITS_CVRD["cvrd"] = {
    "cvrd_train": ("/media/magus/NAS/dataset/MSCOCO/homecoco/train2017","../data/gvrd/class_dict.json","../data/gvrd/relation_dict.json","../data/gvrd/train/images_dict.json","../data/gvrd/train/images_triplets_dict.json"),
    "cvrd_test": ("/media/magus/NAS/dataset/MSCOCO/homecoco/train2017","../data/gvrd/class_dict.json","../data/gvrd/relation_dict.json","../data/gvrd/test/images_dict.json","../data/gvrd/test/images_triplets_dict.json"),
    # "cvrd_test5": ("../data/images","../data/gvrd/class_dict.json","../data/gvrd/relation_dict.json","../data/gvrd/5_images_dict.json","../data/gvrd/5_images_triplets_dict.json"),

    # "cvrd_minitrain": ("../data/images","../data/gvrd/class_dict.json","../data/gvrd/relation_dict.json","../data/gvrd/mini/minitrain_images_dict.json","../data/gvrd/mini/mini_train_images_triplets_dict.json"),
    # "cvrd_minival": ("../data/images","../data/gvrd/class_dict.json","../data/gvrd/relation_dict.json","../data/gvrd/mini/mini_test_images_dict.json","../data/gvrd/mini/mini_test_images_triplets_dict.json")
}
_PREDEFINED_SPLITS_CVRD_DUPLICATED={}
_PREDEFINED_SPLITS_CVRD_DUPLICATED["cvrd_duplicated"] = {
    "cvrd_duplicated_train": ("/media/magus/Data/coco","../data/gvrd/class_dict.json","../data/gvrd/relation_dict.json","../data/gvrd/train/images_dict.json","../data/gvrd/train/images_triplets_dict.json"),
    "cvrd_duplicated_test": ("/media/magus/Data/coco","../data/gvrd/class_dict.json","../data/gvrd/relation_dict.json","../data/gvrd/test/images_dict.json","../data/gvrd/test/images_triplets_dict.json"),
    # "cvrd_test5": ("../data/images","../data/gvrd/class_dict.json","../data/gvrd/relation_dict.json","../data/gvrd/5_images_dict.json","../data/gvrd/5_images_triplets_dict.json"),

    # "cvrd_minitrain": ("../data/images","../data/gvrd/class_dict.json","../data/gvrd/relation_dict.json","../data/gvrd/mini/minitrain_images_dict.json","../data/gvrd/mini/mini_train_images_triplets_dict.json"),
    # "cvrd_minival": ("../data/images","../data/gvrd/class_dict.json","../data/gvrd/relation_dict.json","../data/gvrd/mini/mini_test_images_dict.json","../data/gvrd/mini/mini_test_images_triplets_dict.json")
}

def register_all_cvrd():
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_CVRD.items():
        for key, (image_path,class_json_file,relation_json_file,instance_json_file,triplet_json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_cvrd(
                key,
                _get_builtin_metadata(dataset_name),
                image_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file
            )

def register_all_cvrd_duplicated():
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_CVRD_DUPLICATED.items():
        for key, (image_path,class_json_file,relation_json_file,instance_json_file,triplet_json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_cvrd_duplicated(
                key,
                _get_builtin_metadata(dataset_name),
                image_path,
                class_json_file,
                relation_json_file,
                instance_json_file,
                triplet_json_file
            )

register_all_cvrd()
register_all_cvrd_duplicated()
