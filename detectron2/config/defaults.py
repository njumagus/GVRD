# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2
_C.LESS = False
_C.SIMILARITY= [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]

_C.MODEL = CN()
_C.MODEL.TRAINABLE=[]
_C.MODEL.LOAD_PROPOSALS = False
_C.MODEL.MASK_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

# Path (possibly with schema like catalog:// or detectron2://) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

# -----------------------------------------------------------------------------
# CENTERTRACK
# -----------------------------------------------------------------------------
_C.MODEL.CENTERTRACK = CN()
_C.MODEL.CENTERTRACK.MODEL_PATH = "./centertrack_model/model_epoch_70.pth"
_C.MODEL.CENTERTRACK.SIZE_DIVISIBILITY = 2
_C.MODEL.CENTERTRACK.IN_FEATURES = ["level0", "level1", "level2", "level3", "level4", "level5"]
_C.MODEL.CENTERTRACK.STRIDES = CN({"level0":1,"level1":2,"level2":2,"level3":2,"level4":2,"level5":2})
_C.MODEL.CENTERTRACK.CHANNELS = CN({"level0":16,"level1":32,"level2":64,"level3":128,"level4":256,"level5":512})

_C.MODEL.CENTERTRACK.RESDCN_OUT_STRIDES = [4,8,16,32]
_C.MODEL.CENTERTRACK.RESDCN_OUT_CHANNELS = [256,512,1024,2048]

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)
# Sample size of smallest side by choice or random selection from range give by
# INPUT.MIN_SIZE_TRAIN
_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333

# `True` if cropping is used for data augmentation during training
_C.INPUT.CROP = CN({"ENABLED": False})
# Cropping type:
# - "relative" crop (H * CROP.SIZE[0], W * CROP.SIZE[1]) part of an input of size (H, W)
# - "relative_range" uniformly sample relative crop size from between [CROP.SIZE[0], [CROP.SIZE[1]].
#   and  [1, 1] and use it as in "relative" scenario.
# - "absolute" crop part of an input with absolute size: (CROP.SIZE[0], CROP.SIZE[1]).
_C.INPUT.CROP.TYPE = "relative_range"
# Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
# pixels if CROP.TYPE is "absolute"
_C.INPUT.CROP.SIZE = [0.9, 0.9]


# Whether the model needs RGB, YUV, HSV etc.
# Should be one of the modes defined here, as we use PIL to read the image:
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
# with BGR being the one exception. One can set image format to BGR, we will
# internally use RGB for conversion and flip the channels over
_C.INPUT.FORMAT = "BGR"
# The ground truth mask format that the model will use.
# Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
_C.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
_C.DATASETS.TRAIN = ()
# List of the pre-computed proposal files for training, which must be consistent
# with datasets listed in DATASETS.TRAIN.
_C.DATASETS.PROPOSAL_FILES_TRAIN = ()
# Number of top scoring precomputed proposals to keep for training
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ()
# List of the pre-computed proposal files for test, which must be consistent
# with datasets listed in DATASETS.TEST.
_C.DATASETS.PROPOSAL_FILES_TEST = ()
# Number of top scoring precomputed proposals to keep for test
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
# Options: TrainingSampler, RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# Repeat threshold for RepeatFactorTrainingSampler
_C.DATALOADER.REPEAT_THRESHOLD = 0.0
# if True, the dataloader will filter out images that have no associated
# annotations at train time.
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_AT = 2


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
# Names of the input feature maps to be used by FPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.FPN.IN_FEATURES = []
_C.MODEL.FPN.OUT_CHANNELS = 256

# Options: "" (no norm), "GN"
_C.MODEL.FPN.NORM = ""

# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
_C.MODEL.FPN.FUSE_TYPE = "sum"


# ---------------------------------------------------------------------------- #
# Proposal generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.PROPOSAL_GENERATOR = CN()
# Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
_C.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
# Proposal height and width both need to be greater than MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0

_C.MODEL.CROWD_PROPOSAL_GENERATOR = CN()
# Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
_C.MODEL.CROWD_PROPOSAL_GENERATOR.NAME = "RPN"
# Proposal height and width both need to be greater than MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.CROWD_PROPOSAL_GENERATOR.MIN_SIZE = 0

_C.MODEL.PHRASE_PROPOSAL_GENERATOR = CN()
# Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
_C.MODEL.PHRASE_PROPOSAL_GENERATOR.NAME = "RPN"
# Proposal height and width both need to be greater than MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.PHRASE_PROPOSAL_GENERATOR.MIN_SIZE = 0
# ---------------------------------------------------------------------------- #
# Anchor generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.ANCHOR_GENERATOR = CN()
# The generator can be any name in the ANCHOR_GENERATOR registry
_C.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
# anchor sizes given in absolute pixels w.r.t. the scaled network input.
# Format: list of lists of sizes. SIZES[i] specifies the list of sizes
# to use for IN_FEATURES[i]; len(SIZES) == len(IN_FEATURES) must be true,
# or len(SIZES) == 1 is true and size list SIZES[0] is used for all
# IN_FEATURES.
_C.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
# Anchor aspect ratios.
# Format is list of lists of sizes. ASPECT_RATIOS[i] specifies the list of aspect ratios
# to use for IN_FEATURES[i]; len(ASPECT_RATIOS) == len(IN_FEATURES) must be true,
# or len(ASPECT_RATIOS) == 1 is true and aspect ratio list ASPECT_RATIOS[0] is used
# for all IN_FEATURES.
_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
# Anchor angles.
# list[float], the angle in degrees, for each input feature map.
# ANGLES[i] specifies the list of angles for IN_FEATURES[i].
_C.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]
# Relative offset between the center of the first anchor and the top-left corner of the image
# Units: fraction of feature map stride (e.g., 0.5 means half stride)
# Allowed values are floats in [0, 1) range inclusive.
# Recommended value is 0.5, although it is not expected to affect model accuracy.
_C.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  # used by RPN_HEAD_REGISTRY
_C.MODEL.RPN.USE_GT_INSTANCE = True
# Names of the input feature maps to be used by RPN
# e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
_C.MODEL.RPN.IN_FEATURES = ["res4"]
# Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.BOUNDARY_THRESH = -1
# IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example: 1)
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example: 0)
# Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
# are ignored (-1)
_C.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
_C.MODEL.RPN.IOU_LABELS = [0, -1, 1]
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
_C.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
_C.MODEL.RPN.SMOOTH_L1_BETA = 0.0
_C.MODEL.RPN.LOSS_WEIGHT = 1.0
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
# When FPN is used, this limit is applied per level and then again to the union
# of proposals from all levels
# NOTE: When FPN is used, the meaning of this config is different from Detectron1.
# It means per-batch topk in Detectron1, but per-image topk here.
# See "modeling/rpn/rpn_outputs.py" for details.
_C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
_C.MODEL.RPN.CROWD_PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.RPN.CROWD_PRE_NMS_TOPK_TEST = 6000
_C.MODEL.RPN.CROWD_POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.RPN.CROWD_POST_NMS_TOPK_TEST = 1000
_C.MODEL.RPN.PHRASE_PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.RPN.PHRASE_PRE_NMS_TOPK_TEST = 6000
_C.MODEL.RPN.PHRASE_POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.RPN.PHRASE_POST_NMS_TOPK_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7



# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.CROWD_ROI_HEADS = CN()
_C.MODEL.CROWD_ROI_HEADS.NAME = "CrowdROIHeads"
_C.MODEL.PHRASE_ROI_HEADS = CN()
_C.MODEL.PHRASE_ROI_HEADS.NAME = "PhraseROIHeads"
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
_C.MODEL.ROI_HEADS.LOCATION_DIM=4
_C.MODEL.ROI_HEADS.ENCODE_DIM=0
_C.MODEL.ROI_HEADS.ENCODE_DIM_OUT=1024
_C.MODEL.ROI_HEADS.INSTANCE_HEAD=False
# Number of foreground classes
_C.MODEL.ROI_HEADS.NUM_CLASSES = 80
_C.MODEL.ROI_HEADS.RELATION_NUM_CLASSES = 96
_C.MODEL.ROI_HEADS.POSITIVE_CROWD_THRESHOLD = 0.6
_C.MODEL.ROI_HEADS.NEGATIVE_CROWD_THRESHOLD = 0.5
_C.MODEL.ROI_HEADS.POSITIVE_BOX_THRESHOLD = 1.0
_C.MODEL.ROI_HEADS.WHICH_GT_MATCHED = "all" # "max" matching
# Names of the input feature maps to be used by ROI heads
# Currently all heads (box, mask, ...) use the same input feature map list
# e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
_C.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
_C.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 16 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
# A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
# inference.
_C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
# If True, augment proposals with ground-truth boxes before sampling proposals to
# train ROI heads.
_C.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
_C.MODEL.ROI_HEADS.GT_SIMILARITY_MODE = "support" # "support_normal" "confidence"
_C.MODEL.ROI_HEADS.SIMILARITY_MODE = "att"
_C.MODEL.ROI_HEADS.SIMILARITY_FEATURE = "loc"
_C.MODEL.ROI_HEADS.SIMILARITY_LOSS = "f1"
_C.MODEL.ROI_HEADS.SIMILARITY_TRAIN_ONLY_SAME_CLASS = False
_C.MODEL.ROI_HEADS.GRAPH_SEARCH_MODE = ["connect"] # "complex" "both"
_C.MODEL.ROI_HEADS.BIAS=False
# ---------------------------------------------------------------------------- #
# Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_HEAD = CN()
# C4 don't use head name option
# Options for non-C4 models: FastRCNNConvFCHead,
_C.MODEL.ROI_BOX_HEAD.NAME = ""
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
_C.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

_C.MODEL.ROI_BOX_HEAD.NUM_FC = 0
# Hidden layer dimension for FC layers in the RoI box head
_C.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
_C.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
# Channel dimension for Conv layers in the RoI box head
_C.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
_C.MODEL.ROI_BOX_HEAD.NORM = ""
# Whether to use class agnostic for bbox regression
_C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
# If true, RoI heads use bounding boxes predicted by the box head rather than proposal boxes.
_C.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False

# ---------------------------------------------------------------------------- #
# Cascaded Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_CASCADE_HEAD = CN()
# The number of cascade stages is implicitly defined by the length of the following two configs.
_C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)
_C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)


# ---------------------------------------------------------------------------- #
# Mask Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.NUM_CONV = 0  # The number of convs in the mask head
_C.MODEL.ROI_MASK_HEAD.CONV_DIM = 256
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
_C.MODEL.ROI_MASK_HEAD.NORM = ""
# Whether to use class agnostic for mask prediction
_C.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"

# ---------------------------------------------------------------------------- #
# Heatmap Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEATMAP_HEAD = CN()
_C.MODEL.ROI_HEATMAP_HEAD.NAME = "MaskRCNNConvUpsampleHead"
_C.MODEL.ROI_HEATMAP_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_HEATMAP_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_HEATMAP_HEAD.NUM_CONV = 0  # The number of convs in the mask head
_C.MODEL.ROI_HEATMAP_HEAD.CONV_DIM = 256
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
_C.MODEL.ROI_HEATMAP_HEAD.NORM = ""
# Whether to use class agnostic for mask prediction
_C.MODEL.ROI_HEATMAP_HEAD.CLS_AGNOSTIC_MASK = True
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_HEATMAP_HEAD.POOLER_TYPE = "ROIAlignV2"

# ---------------------------------------------------------------------------- #
# Semantic Segmentation Head
# ---------------------------------------------------------------------------- #
_C.MODEL.SEM_SEG_HEAD = CN()
_C.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"
_C.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
# Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
# the correposnding pixel.
_C.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
# Number of classes in the semantic segmentation head
_C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 54
# Number of channels in the 3x3 convs inside semantic-FPN heads.
_C.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128
# Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
_C.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
# Normalization method for the convolution layers. Options: "" (no norm), "GN".
_C.MODEL.SEM_SEG_HEAD.NORM = "GN"
_C.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0

_C.MODEL.PANOPTIC_FPN = CN()
# Scaling of all losses from instance detection / segmentation head.
_C.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT = 1.0

# options when combining instance & semantic segmentation outputs
_C.MODEL.PANOPTIC_FPN.COMBINE = CN({"ENABLED": True})
_C.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5
_C.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 4096
_C.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5


# ---------------------------------------------------------------------------- #
# RetinaNet Head
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# This is the number of foreground classes.
_C.MODEL.RETINANET.NUM_CLASSES = 80

_C.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# IoU overlap ratio [bg, fg] for labeling anchors.
# Anchors with < bg are labeled negative (0)
# Anchors  with >= bg and < fg are ignored (-1)
# Anchors with >= fg are labeled positive (1)
_C.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
_C.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
_C.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
_C.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.RETINANET.NMS_THRESH_TEST = 0.5

# Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
_C.MODEL.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Loss parameters
_C.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1


# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

_C.MODEL.RESNETS.DEPTH = 50
_C.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # res4 for C4 backbone, res2..5 for FPN backbone

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.RESNETS.NORM = "FrozenBN"

# Baseline width of each group.
# Scaling this parameters will scale the width of all bottleneck layers.
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

# Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# Apply Deformable Convolution in stages
# Specify if apply deform_conv on Res2, Res3, Res4, Res5
_C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
# Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
# Use False for DeformableV1.
_C.MODEL.RESNETS.DEFORM_MODULATED = False
# Number of groups in deformable conv.
_C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1

# ---------------------------------------------------------------------------- #
# Relation
# ---------------------------------------------------------------------------- #
_C.MODEL.PREDICATE_HEADS = CN()
_C.MODEL.PREDICATE_HEADS.RELATION_NUM = 250
_C.MODEL.PREDICATE_HEADS.INSTANCE_NUM = 134
_C.MODEL.PREDICATE_HEADS.NAME = "PredicateHeads"
_C.MODEL.PREDICATE_HEADS.LANGUAGE_HEAD = CN()
_C.MODEL.PREDICATE_HEADS.LANGUAGE_HEAD.NAME = "LanguageHead"
_C.MODEL.PREDICATE_HEADS.VISUAL_HEAD = CN()
_C.MODEL.PREDICATE_HEADS.VISUAL_HEAD.NAME = "LanguageHead"

_C.MODEL.RELATION_HEADS = CN()
_C.MODEL.RELATION_HEADS.IMAGE_SIZE=800
_C.MODEL.RELATION_HEADS.INSTANCE_NUM=134
_C.MODEL.RELATION_HEADS.RELATION_NUM=250
_C.MODEL.RELATION_HEADS.NAME="StandardRelationHeads"
_C.MODEL.RELATION_HEADS.SMOOTH_L1_BETA = 0.0
_C.MODEL.RELATION_HEADS.MASK_ON = False
_C.MODEL.RELATION_HEADS.RELATION_HEAD_LIST=["instance"]
_C.MODEL.RELATION_HEADS.FEATURE_DIM=512
_C.MODEL.RELATION_HEADS.FEATURE_INIT="normal_constant"
_C.MODEL.RELATION_HEADS.ATTENTION=True
_C.MODEL.RELATION_HEADS.TRAIN_RELATIVE=False
_C.MODEL.RELATION_HEADS.RELATIVE=False
_C.MODEL.RELATION_HEADS.REWEIGHT=0.0
_C.MODEL.RELATION_HEADS.REWEIGHT_AFTER=False
_C.MODEL.RELATION_HEADS.IN_CROWD_USE_CROWD=False
_C.MODEL.RELATION_HEADS.USE_ATTENTION_FEATURE=False
_C.MODEL.RELATION_HEADS.MINUS=False
_C.MODEL.RELATION_HEADS.CROWD_LOSS="bce"

# ---------------------------------------------------------------------------- #
# DETR
# ---------------------------------------------------------------------------- #
_C.MODEL.DETR = CN()
_C.MODEL.DETR.NUM_CLASSES = 80
# For Segmentation
_C.MODEL.DETR.FROZEN_WEIGHTS = ''
# LOSS
_C.MODEL.DETR.GIOU_WEIGHT = 2.0
_C.MODEL.DETR.L1_WEIGHT = 5.0
_C.MODEL.DETR.DEEP_SUPERVISION = True
_C.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1
# TRANSFORMER
_C.MODEL.DETR.NHEADS = 8
_C.MODEL.DETR.DROPOUT = 0.1
_C.MODEL.DETR.DIM_FEEDFORWARD = 2048
_C.MODEL.DETR.ENC_LAYERS = 6
_C.MODEL.DETR.DEC_LAYERS = 6
_C.MODEL.DETR.PRE_NORM = False
_C.MODEL.DETR.HIDDEN_DIM = 256
_C.MODEL.DETR.NUM_OBJECT_QUERIES = 100

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.OPTIMIZER="SGD"
_C.SOLVER.BACKBONE_MULTIPLIER=0.1
# See detectron2/solver/build.py for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of images per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
_C.SOLVER.IMS_PER_BATCH = 16

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY
# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 0
# Maximum number of detections to return per image during inference (100 is
# based on the limit established for the COCO dataset).
_C.TEST.DETECTIONS_PER_IMAGE = 100

_C.TEST.AUG = CN({"ENABLED": False})
_C.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
_C.TEST.AUG.MAX_SIZE = 4000
_C.TEST.AUG.FLIP = True

_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed does not
# guarantee fully deterministic behavior.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from detectron2.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0
