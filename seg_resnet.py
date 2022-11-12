from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time
import os
import copy
from math import floor
import cv2


# cudnn.benchmark = True
plt.ion()   # interactive mode

gun_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/t/tianqi/CS4243_proj/my_utils/best1110.pt')
gun_model.conf = 0.1
knife_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/t/tianqi/CS4243_proj/my_utils/knife_best.pt')
knife_model.conf = 0.1

def get_classifcation_bounding_box(file_path, model, asize):
    results = model(file_path, size=asize)

    objects = []
    for obj in results.xyxy[0]:
        objects.append(
            {
                "class": int(obj[5]), 
                "xmin": float(obj[0]),
                "ymin": float(obj[1]),
                "xmax": float(obj[2]),
                "ymax": float(obj[3]),
            })
    return objects

def detect_overlap(mask, xmin, xmax, ymin, ymax):
    return np.count_nonzero(mask[ymin:ymax, xmin:xmax]) > 0

def get_seg_file(img_path):
    paths = img_path.split('/')
    paths[0] = "seg"
    seg_dir = '/'.join(paths)
    im_read = cv2.imread(seg_dir, 0)
    return im_read

def get_seg_bin(img_path, thres=50):
    seg = get_seg_file(img_path)
    ret, mask = cv2.threshold(seg, thres, 1, cv2.THRESH_BINARY)
    return mask

count = 0
def combine_mask_bounding_box(file_path, mask_thres, box_thres):
    global count
    gun_boxes = get_classifcation_bounding_box(file_path, gun_model, 540)
    knife_boxes = get_classifcation_bounding_box(file_path, knife_model, 640)
    count += len(knife_boxes)

    boxes = gun_boxes + knife_boxes
    mask = get_seg_bin(file_path, thres=mask_thres)
    r, c = mask.shape

    for box in boxes:
        xmin, xmax, ymin, ymax = floor(box["xmin"]), floor(box["xmax"]), floor(box["ymin"]), floor(box["ymax"])
        if detect_overlap(mask, xmin, xmax, ymin, ymax):
            mask[max(0, ymin-box_thres):min(ymax+box_thres, r), max(0, xmin-box_thres):min(xmax+box_thres, c)] = 1
    return mask

def mask_img_with_box(img, img_path):
    mask = combine_mask_bounding_box(img_path, 0, 0.1)
    seg_img = cv2.bitwise_and(img, img, mask = mask)
    return seg_img

# class SegDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.segs = []
#         self.labels = []
#         self.__preprocess__()


#     def __preprocess__(self):
#         subfolders = ['carrying', 'normal', 'threat']
#         for i in range(len(subfolders)):
#             # print(subfolders[i])
#             files = os.listdir(os.path.join(self.root_dir, subfolders[i]))
#             for f in files:
#                 img_path = os.path.join(self.root_dir, subfolders[i], f)
#                 mask = combine_mask_bounding_box(img_path, 0, 50)
#                 img = cv2.imread(img_path)

#                 seg_img = cv2.bitwise_and(img, img, mask = mask)

#                 self.segs.append(seg_img)
#                 self.labels.append(i)

#     def __len__(self):
#         return len(self.segs)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         image = self.segs[idx]
#         label = self.labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# data_transforms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])


# data_dir = 'data'
# image_datasets = {x: SegDataset(os.path.join(data_dir, x), transform=data_transforms)
#                   for x in ['train', 'val', 'test']}


# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                              shuffle=True, num_workers=4)
#               for x in ['train', 'val', 'test']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
# # class_names = image_datasets['train'].classes

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

