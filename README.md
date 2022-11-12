# Dataset

All images are resized to 640 x 640 pixels and split into 3 different sets, train, val and test sets. 

```
.
├── data                 
│   ├── test   
│   ├── train      
│   └── val        
├── seg         # contains the segmentation masks of the dataset              
│   ├── test   
│   ├── train      
│   └── val    
└── ...
```

# Baseline

We tried 2 different Resnets, Resnet18 and Resnet50 as our baseline. The code can be found in

* [`baseline_resnet.ipynb`](baseline_resnet.ipynb) (Resnet18)
* [`baseline_resnet_50.ipynb`](baseline_resnet_50.ipynb) (Resnet50)

# Resnet with segmentation masks

We tried to combine the images with its segmentation masks and yolov5 object bounding boxes to reduce the amount of background noises in the images.

The segmentation masks were generated with [`infer_seg1.py`](https://github.com/CS4243-Project-ResNet/utils/blob/main/infer_seg1.py).

The code can be found in [`seg_resnet.ipynb`](seg_resnet.ipynb).
