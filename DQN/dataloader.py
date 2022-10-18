from cgi import test
from monai.utils import first, set_determinism
from monai.config import KeysCollection
from monai.transforms import (
    Transform,
    MapTransform,
    Randomizable,
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
from monai.transforms.utils import generate_spatial_bounding_box
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import random
import numpy as np

from config import config


class SelectTargetClass(MapTransform):
    def __init__(self, keys: KeysCollection, target_class):
        super().__init__(keys)
        self.keys = keys
        self.target_class = target_class

    def select(self, data):
        data[data != self.target_class] = 0
        data[data == self.target_class] = 1

        return data
        
    def __call__(self, img):
        d = dict(img)
        image, label = d['image'], d['label']
        seg_label = torch.clone(label)
        for key in self.key_iterator(d):
            label = self.select(d[key])
        return {'image': image, 'label': label, "seg_label": seg_label}


class GetBoundingBox(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        self.keys = keys

    def select(self, data):
        if len(data.size()) == 3:
          data = torch.unsqueeze(data, dim=0)

        box = generate_spatial_bounding_box(data)
        return box
        
    def __call__(self, img):
        d = dict(img)
        for key in self.key_iterator(d):
            d[key] = self.select(d[key])
        return d


class CT_DataLoader():
    def __init__(self, mode="train"):

        self.mode = config.model
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size

        self.folder_name = "Training" if self.mode == "train" else "Testing"

        self.images = sorted(glob.glob(os.path.join(self.data_dir, self.folder_name, "img", "*.nii.gz")))
        self.labels = sorted(glob.glob(os.path.join(self.data_dir, self.folder_name, "label", "*.nii.gz"))) if self.mode == "train" else []

        self.data_size = len(self.images)

        self.data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(self.images, self.labels)
        ]

        self.transformed_data = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    # TODO: scale image to same size
                    ScaleIntensityRanged(
                        keys=["image"], a_min=-57, a_max=164,
                        b_min=0.0, b_max=1.0, clip=True,
                    ),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(keys=["image", "label"], pixdim=(
                        1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                    SelectTargetClass(keys=["label"], target_class = 6), 
                    GetBoundingBox(keys=["label"])
                ]
            )

        self.dataset = Dataset(data=self.data_dict, transform=self.transformed_data)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)


    def sample_scan_circular(self):
        '''
        return a  CT scan data from the dataloader
        '''
        image, label = next(iter(self.dataloader))

        return image[0][0], label[0][0]
