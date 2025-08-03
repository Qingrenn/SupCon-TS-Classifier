# SupCon-TS-Classifier
Supervised Contrastive Learning for Time Series Classification

## Dataset Structure

Organize training datasets following the below strucutre:

```bash
datasets/train_datasets
├── gw
│   ├── cbc
│   └── gn
├── leaves
│   ├── CEP_LCs 
│   ├── DSCT_LCs
│   ├── EB_LCs
│   ├── LPV_LCs
│   ├── Non-var_LCs
│   ├── ROT_LCs
│   └── RR_LCs
├── sleep
│   ├── class0
│   ├── class1
│   ├── class2
│   ├── class3
│   └── class4
└── stead
    ├── noise
    └── seism
```

The test or validation dataset needs to be organized according to the above structure.

```bash
datasets/test_datasets
...

datasets/val_datasets
...
```

## Training

Training with Contrastive Learning: the specific configs can be found in `train.py`
```bash
python train.py 
python -m torch.distributed.run --nproc_per_node=8 train.py
```

Training with Classification Head: the specific configs can be found in `train.py`
```bash
python train_class.py 
python -m torch.distributed.run --nproc_per_node=8 train_class.py
```

## Test

Test model with Classification Head
```
python test_classifier.py
```

Test model via embedding-based KNN classifier
```
python test_knn.py
```

