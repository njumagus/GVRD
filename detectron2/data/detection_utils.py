# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Common data processing utilities that are used in a
typical object detection data pipeline.
"""
import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.utils.torch_utils import union_box
from fvcore.common.file_io import PathManager
from PIL import Image, ImageOps

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Triplets,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

from . import transforms as T
from .catalog import MetadataCatalog


class SizeMismatchError(ValueError):
    """
    When loaded image has difference width/height compared with annotation.
    """


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"

    Returns:
        image (np.ndarray): an HWC image in the given format.
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise SizeMismatchError(
                "Mismatched (W,H){}, got {}, expect {}".format(
                    " for image " + dataset_dict["file_name"]
                    if "file_name" in dataset_dict
                    else "",
                    image_wh,
                    expected_wh,
                )
            )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]


def transform_proposals(dataset_dict, image_shape, transforms, min_box_side_len, proposal_topk):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        min_box_side_len (int): keep proposals with at least this size
        proposal_topk (int): only keep top-K scoring proposals

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "proposal_boxes" in dataset_dict:
        # Transform proposal boxes
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)
        objectness_logits = torch.as_tensor(
            dataset_dict.pop("proposal_objectness_logits").astype("float32")
        )

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_side_len)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        dataset_dict["proposals"] = proposals


def transform_instance_annotations(
    annotation, transforms, image_size
):
    """
    Apply transforms to box, segmentation annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_box([bbox])[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        elif isinstance(segm, np.ndarray):
            mask = transforms.apply_segmentation(segm)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    return annotation


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """

    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    # class_names = [obj["category_name"] for obj in annos]
    # target.gt_class_names = class_names

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    return target


def triplets_to_triplets(triplets_list, image_size):
    target = Triplets(image_size)

    gt_subject_ids_list = torch.tensor([triplet['subject_ids'] for triplet in triplets_list], dtype=torch.int64)
    gt_object_ids_list = torch.tensor([triplet['object_ids'] for triplet in triplets_list], dtype=torch.int64)
    gt_relation_onehot = torch.tensor([triplet['relation_onehot'] for triplet in triplets_list], dtype=torch.int64)

    target.gt_subject_ids_list = gt_subject_ids_list
    target.gt_object_ids_list = gt_object_ids_list
    target.gt_relation_onehot = gt_relation_onehot

    return target

def triplets_to_crowds_phrases(instances_list, triplets_list, image_size):
    crowds = Instances(image_size)
    phrases = Instances(image_size)

    crowd_instance_ids_str_list = []
    crowd_instance_ids_list = []
    crowd_classes = []
    crowd_boxes = []
    crowd_masks = []

    sub_crowd_ids = []
    obj_crowd_ids = []
    phrase_boxes=[]
    phrase_masks=[]
    phrase_relation_onehots = []
    for i in range(len(triplets_list)):
        sub_box = torch.Tensor()
        sub_seg = torch.zeros(image_size)
        sub_cls = torch.Tensor()
        sub_ids = ""
        for sub_id in torch.sort(triplets_list.gt_subject_ids_list[i])[0]:
            if sub_id==-1:
                continue
            sub_ids += str(sub_id.item()).zfill(2)
        if sub_ids not in crowd_instance_ids_str_list:
            crowd_instance_ids_str_list.append(sub_ids)
            crowd_instance_ids_list.append(triplets_list.gt_subject_ids_list[i])
            for sub_id in triplets_list.gt_subject_ids_list[i]:
                if sub_id==-1:
                    continue
                box = instances_list.gt_boxes.tensor[sub_id]
                seg = instances_list.gt_masks.tensor[sub_id]
                if sub_box.shape[0] == 0:
                    sub_box = box
                    sub_seg = seg
                    sub_cls = instances_list.gt_classes[sub_id]
                else:
                    sub_box = union_box(box, sub_box)
                    sub_seg = torch.max(seg, sub_seg)
            crowd_classes.append(sub_cls)
            crowd_boxes.append(sub_box)
            crowd_masks.append(sub_seg)
        sub_crowd_ids.append(crowd_instance_ids_str_list.index(sub_ids))

        obj_box = torch.Tensor()
        obj_seg = torch.zeros(image_size)
        obj_cls = torch.Tensor()
        obj_ids = ""
        for obj_id in torch.sort(triplets_list.gt_object_ids_list[i])[0]:
            if obj_id == -1:
                continue
            obj_ids += str(obj_id.item()).zfill(2)
        if obj_ids not in crowd_instance_ids_str_list:
            crowd_instance_ids_str_list.append(obj_ids)
            crowd_instance_ids_list.append(triplets_list.gt_object_ids_list[i])
            for obj_id in triplets_list.gt_object_ids_list[i]:
                if obj_id==-1:
                    continue
                box = instances_list.gt_boxes.tensor[obj_id]
                seg = instances_list.gt_masks.tensor[obj_id]
                obj_cls = instances_list.gt_classes[obj_id]
                if obj_box.shape[0] == 0:
                    obj_box = box
                    obj_seg = seg
                else:
                    obj_box = union_box(box, obj_box)
                    obj_seg = torch.max(seg, obj_seg)
            crowd_classes.append(obj_cls)
            crowd_boxes.append(obj_box)
            crowd_masks.append(obj_seg)
        obj_crowd_ids.append(crowd_instance_ids_str_list.index(obj_ids))
        phrase_boxes.append(union_box(crowd_boxes[sub_crowd_ids[-1]],crowd_boxes[obj_crowd_ids[-1]]))
        phrase_masks.append(torch.max(crowd_masks[sub_crowd_ids[-1]],crowd_masks[obj_crowd_ids[-1]]))
        phrase_relation_onehots.append(triplets_list.gt_relation_onehot[i])

    crowds.gt_instance_ids_list=torch.stack(crowd_instance_ids_list)
    crowds.gt_instance_lens=torch.sum(crowds.gt_instance_ids_list>=0,dim=1)
    crowds.gt_classes=torch.stack(crowd_classes)
    crowds.gt_boxes=Boxes(torch.stack(crowd_boxes))
    crowds.gt_masks=BitMasks(torch.stack(crowd_masks))

    phrases.gt_subject_ids_list = triplets_list.gt_subject_ids_list
    phrases.gt_object_ids_list = triplets_list.gt_object_ids_list
    phrases.gt_subject_lens = torch.sum(phrases.gt_subject_ids_list >= 0, dim=1)
    phrases.gt_object_lens = torch.sum(phrases.gt_object_ids_list >= 0, dim=1)
    phrases.gt_subject_crowd_ids = torch.Tensor(sub_crowd_ids).long()
    phrases.gt_object_crowd_ids = torch.Tensor(obj_crowd_ids).long()
    phrases.gt_subject_classes = crowds.gt_classes[phrases.gt_subject_crowd_ids.long()]
    phrases.gt_object_classes = crowds.gt_classes[phrases.gt_object_crowd_ids.long()]
    phrases.gt_subject_masks = crowds.gt_masks[phrases.gt_subject_crowd_ids.long()]
    phrases.gt_object_masks = crowds.gt_masks[phrases.gt_object_crowd_ids.long()]
    phrases.gt_boxes = Boxes(torch.stack(phrase_boxes))
    phrases.gt_masks = BitMasks(torch.stack(phrase_masks))
    phrases.gt_relation_onehots = torch.stack(phrase_relation_onehots)

    return crowds, phrases


def annotations_to_instances_rotated(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Compared to `annotations_to_instances`, this function is for rotated boxes only

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            Containing fields "gt_boxes", "gt_classes",
            if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [obj["bbox"] for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = RotatedBoxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    return target


def filter_empty_instances(instances, by_box=True, by_mask=True):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty())
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return instances[m]

def gen_crop_transform_with_instance(crop_size, image_size, instance):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    crop_size = np.asarray(crop_size, dtype=np.int32)
    bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
    return T.CropTransform(x0, y0, crop_size[1], crop_size[0])


def check_metadata_consistency(key, dataset_names):
    """
    Check that the datasets have consistent metadata.

    Args:
        key (str): a metadata key
        dataset_names (list[str]): a list of dataset names

    Raises:
        AttributeError: if the key does not exist in the metadata
        ValueError: if the given datasets do not have the same metadata values defined by key
    """
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(key, dataset_names[idx], str(entry))
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(
                    key, dataset_names[0], str(entries_per_dataset[0])
                )
            )
            raise ValueError("Datasets have different metadata '{}'!".format(key))

def bulid_square_transform_gen(cfg, is_train):
    if is_train:
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        sample_style = "choice"
    tfm_gens = []
    tfm_gens.append(T.Resize((cfg.MODEL.RELATION_HEADS.IMAGE_SIZE, cfg.MODEL.RELATION_HEADS.IMAGE_SIZE)))
    return tfm_gens

def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    # if is_train:
    #     tfm_gens.append(T.RandomFlip())
    #     logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens
