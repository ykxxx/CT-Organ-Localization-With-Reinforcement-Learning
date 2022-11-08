from cgi import test
from monai.utils import first, set_determinism
from monai.config import KeysCollection
from monai.transforms import (
    Transform,
    MapTransform,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
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
import os
import glob
import random
import numpy as np

from config import config


class ApplyWindowing(MapTransform):
  def __init__(self, keys: KeysCollection, window_center, window_width):
      super().__init__(keys)
      self.window_center = window_center
      self.window_width = window_width
      self.pixel_low = self.window_center - self.window_width / 2
      self.pixel_high = self.window_center + self.window_width / 2
  
  def select(self, data):

      data = torch.where(data < self.pixel_low, self.pixel_low, data)
      data = torch.where(data > self.pixel_high, self.pixel_high, data)

      data_norm = (data - torch.min(data)) / (torch.max(data) - torch.min(data))

      return data_norm

  def __call__(self, img):
      d = dict(img)
      for key in self.key_iterator(d):
          d[key] = self.select(d[key])
      return d

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

        self.mode = mode 
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

        self.transformed_data = Compose([
                    LoadImaged(keys=["image", "label"]), 
                    ApplyWindowing(keys=["image"], window_center=40, window_width=400),
                    SelectTargetClass(keys=["label"], target_class = 6), 
                    GetBoundingBox(keys=["label"]),
                    EnsureChannelFirstd(keys=["image"]),])

        self.dataset = Dataset(data=self.data_dicts, transform=self.transformed_data)
        # self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)


    def sample_circular(self, id = None):
        '''
        return a  CT scan data from the dataloader
        '''
        
        # sample_batch = next(iter(self.dataloader))

        # # batch_size * h * w * d
        # image = sample_batch["image"][0]
        # # list: length 2
        # label = sample_batch["label"]
        data_size = len(self.dataset)
        if id == None:
          idx = np.random.randint(data_size, size=1)[0]
        else:
          idx = id
      
        return self.dataset[idx]["image"], self.dataset[idx]["label"]


def main():
    # Test functionality of CT_DataLoader

    train_loader = CT_DataLoader(mode = "train")
    test_loader = CT_DataLoader(mode = "test")

    # sample_batch = next(iter(train_loader.dataloader))
    image, label= train_loader.sample_circular()

    pr
    int(image.shape)
    print(label)

    fig, ax = plt.subplots()
    ax.imshow(image[0, :, :, 60],cmap='gray', interpolation=None)