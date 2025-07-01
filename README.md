# Group Visual Relation Detection
## Environment
#### main dependencies
- python 3.8
- cuda 11.1
- cudnn 8.0
- pytorch 1.7 
```
apt-get install libglib2.0-dev libsm6 libxrender-dev libxext6  
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html  
```
#### other dependencies
- cython==0.29.21
- matplotlib==3.2.2
- Shapely==1.7.1
- termcolor==1.1.0
- yacs==0.1.8
- opencv_python==3.4.8.29
- cloudpickle==1.5.0
- numpy==1.20.0
- pycocotools==2.0.2
- iopath==0.1.7
- tabulate==0.8.7
- scipy==1.5.0
- Pillow==8.1.0
- setuptools==41.0.0
- tensorboard==2.4.1
- pyyaml==5.4.1

You can just run the script:
```
pip install -r requirements.txt
```  
If you encounter an error when update the pyyaml, try to: 
```
pip install pyyaml==5.4.1 --ignore-installed
``` 

#### directly use the docker
If your GPU support CUDA11.1, you can directly use the docker image that we provide.
```
docker pull yfraquelle/vroid_env:v1  
docker run -it -v /host/path/to/VROID:/docker/path/to/VROID --ipc=host --net=host <image_id> /bin/bash  
```

## Dataset
We constructed a [COCO-GVR](https://drive.google.com/file/d/1MY0jCKdJ5FHx4_VfH-KBCmXQWbsa2LHP/view?usp=sharing) dataset.
Download the data to  
```
../data/gvrd  
├── train  
│   ├── images_dict.json  
│   └── images_triplets_dict.json  
├── test  
│   ├── images_dict.json  
│   └── images_triplets_dict.json  
├── class_dict.json  
└── relation_dict.json  
```
## Preprocess
#### installation
```
python setup.py build develop
```
#### data preparation
Download the [models](https://pan.baidu.com/s/1ohpgwH0tgvgvgK1jre_EGg?pwd=8cgj) to:
```
./  
├── models/R101-FPN-3x.pkl  
└── output  
    ├── relation_clust_allencode_distance_ctrans2  
    │   └── gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=05  
    │       └── model_final.pth    
    └── ...  
```
## Train
```
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=05.yaml"
```

## Test
```
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_05_reweight1after.yaml" --crth 0.5 --merge merge
```

## Evaluate
```
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_05_reweight1after" --crth 0.5 --merge merge
```

## Ablation Study
```
SGRP-VIS: 
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_vis)_relation(crowdphrase)=05.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/test_gvrd_instance(gt_anchor)_group(cosinetrain_vis)_relation(crowdphrase)=less_05_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_vis)_relation(crowdphrase)=less_05_reweight1after" --crth 0.5 --merge merge

SGRP-GPP: 
python main.py --mode train_crowd --config "configs/crowd_proposal_like_instance_anchor.yaml"
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_precrowd_ctrans2/gvrd_instance(gt_anchor)_group(precrowdtrain)_relation(crowdphrase)=05.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_precrowd_ctrans2/test_gvrd_instance(gt_anchor)_group(precrowdtrain)_relation(crowdphrase)=less_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_precrowd_ctrans2/gvrd_instance(gt_anchor)_group(precrowdtrain)_relation(crowdphrase)=less_reweight1after" --crth 0.5 --merge merge

SGRP-PPP: 
python main.py --mode train_phrase --config "configs/phrase_proposal_like_instance_anchor.yaml"
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_prephrase_ctrans2/gvrd_instance(gt_anchor)_group(prephrasetrain)_relation(crowdphrase)=05.yaml"
python main.py --mode test_relation  --config "configs/gvrd/relation_clust_allencode_prephrase_ctrans2/test_gvrd_instance(gt_anchor)_group(prephrasetrain)_relation(crowdphrase)=less_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_prephrase_ctrans2/gvrd_instance(gt_anchor)_group(prephrasetrain)_relation(crowdphrase)=less_reweight1after" --crth 0.5 --merge merge
```
```
SGRP-w/o CA: 
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_trans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=05.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_trans2/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_05_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_trans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_05_reweight1after" --crth 0.5 --merge merge

SGRP-w/o GP: 
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2nogp/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=05.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2nogp/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_05_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_ctrans2nogp/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_05_reweight1after" --crth 0.5 --merge merge

SGRP-BiLSTM: 
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_bilstm/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=05.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_bilstm/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_05_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_bilstm/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_05_reweight1after" --crth 0.5 --merge merge

SGRP-GCN: 
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_gcn/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=05.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_gcn/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_05_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_gcn/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_05_reweight1after" --crth 0.5 --merge merge
```
```
SGRP-w/o G&P: 
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation()=05.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation()=less_05_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation()=less_05_reweight1after" --crth 0.5 --merge merge

SGRP-w/o P: 
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowd)=05.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowd)=less_05_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowd)=less_05_reweight1after" --crth 0.5 --merge merge

SGRP-Mean: 
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(ignoreattention_crowdphrase)=05.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(ignoreattention_crowdphrase)=less_05_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(ignoreattention_crowdphrase)=less_05_reweight1after" --crth 0.5 --merge merge
```
```
SGRP-0
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=0.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_0_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_0_reweight1after" --crth 0.5 --merge merge

SGRP-0369
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=0369.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_0369_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_0369_reweight1after" --crth 0.5 --merge merge

SGRP-0_9
python main.py --mode train_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=0-9.yaml"
python main.py --mode test_relation --config "configs/gvrd/relation_clust_allencode_distance_ctrans2/test_gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_0-9_reweight1after.yaml" --crth 0.5 --merge merge
python cvrd_evaluate.py --func_name "output/relation_clust_allencode_distance_ctrans2/gvrd_instance(gt_anchor)_group(cosinetrain_loc)_relation(crowdphrase)=less_0-9_reweight1after" --crth 0.5 --merge merge
```
