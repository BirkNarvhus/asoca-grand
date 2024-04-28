#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

import monai.data
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImageD, ToTensorD, RandSpatialCropD, CenterSpatialCropD, \
    EnsureChannelFirstd, EnsureTyped, NormalizeIntensityd, RandScaleIntensityd, \
    RandShiftIntensityd, ResizeD
from monai.metrics import HausdorffDistanceMetric, DiceMetric, MeanIoU
from monai.losses import DiceFocalLoss

from glob import glob

from tqdm import tqdm


# In[ ]:


# Load the configuration file
config_dict = {}
try:
    with open("configs.yaml", 'r') as stream:
        config_dict = monai.bundle.utils.yaml.load(stream, Loader=monai.bundle.utils.yaml.FullLoader)
except FileNotFoundError:
    print("Config file not found.")
    exit()


# In[ ]:


class Meter:
    '''factory for storing and updating iou and dice scores.'''

    def __init__(self):
        self.haus_dorf = HausdorffDistanceMetric(include_background=True, percentile=0.95, reduction='mean_batch',
                                                 get_not_nans=False)
        self.dice = DiceMetric(reduction='mean_batch', get_not_nans=False)
        self.iou = MeanIoU(reduction='mean_batch', get_not_nans=False)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Takes: logits from output model and targets,
        calculates dice and iou scores, and stores them in lists.
        """
        logits = torch.nn.Sigmoid()(logits)

        self.haus_dorf(logits, targets)
        self.dice(logits, targets)
        self.iou(logits, targets)

    def get_metrics(self):
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice = self.dice.aggregate().item()
        iou = self.iou.aggregate().item()
        hausdorff = self.haus_dorf.aggregate().item()
        return dice, iou, hausdorff

    def reset(self):
        self.dice.reset()
        self.iou.reset()
        self.haus_dorf.reset()


# In[ ]:


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 train_loader: monai.data.DataLoader,
                 test_loader: monai.data.DataLoader,
                 epochs: int = 100,
                 device: str = 'cuda',
                 plot=False,
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints',
                 output_dir: str = 'outputs',
                 checkpoint_interval: int = 10,
                 lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.epochs = epochs
        self.device = device
        self.model.to(self.device)

        self.plot = plot

        self.losses = {'train': [], 'test': []}
        self.dice_scores = {'train': [], 'test': []}
        self.iou_scores = {'train': [], 'test': []}
        self.hausdorff_scores = {'train': [], 'test': []}

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

        self.checkpoint_interval = checkpoint_interval

        self.best_test_loss = np.inf

        self.meter = Meter()

    def loss_and_logits(self, images: torch.Tensor, masks: torch.Tensor):
        images = images.to(self.device)
        masks = masks.to(self.device)

        logits = self.model(images)
        loss = self.criterion(logits, masks)
        return loss, logits

    def next_epoch(self, epoch, test=False):
        self.model.train() if not test else self.model.eval()
        running_loss = 0.0

        if not test:
            self.optimizer.zero_grad()

        for i, data_dict in enumerate(self.train_loader if not test else self.test_loader):
            images, masks = data_dict['image'], data_dict['mask']
            del data_dict
            loss, logits = self.loss_and_logits(images, masks)

            if not test:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item()
            self.meter.update(logits.detach().cpu(), masks.detach().cpu())

        epoch_loss = running_loss / len(self.train_loader if not test else self.test_loader)
        dice, iou, hausdorff = self.meter.get_metrics()

        self.losses['train' if not test else 'test'].append(epoch_loss)
        self.dice_scores['train' if not test else 'test'].append(dice)
        self.iou_scores['train' if not test else 'test'].append(iou)
        self.hausdorff_scores['train' if not test else 'test'].append(hausdorff)
        self.meter.reset()

        return epoch_loss, (dice, iou, hausdorff)

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            self.next_epoch(epoch, test=False)

            with torch.no_grad():
                test_loss, metrics = self.next_epoch(epoch, test=True)
                self.lr_scheduler.step(test_loss)

            if self.plot:
                self.plot_metrics()

            if test_one:
                print(f"1 epoch test loss: {test_loss:.4f} and metrics: dice - {metrics[0]:.6f} iou - {metrics[1]:.6f}",
                      f"hassdorf - {metrics[2]:.6f}")
                break
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                torch.save(self.model.state_dict(), self.output_dir + "/" + 'best_model.pth')

            if (epoch + 1) % self.checkpoint_interval == 0:
                print(f"Saving checkpoint at epoch: {epoch + 1}, test loss: {test_loss:.4f} and metrics: dice - {metrics[0]:.6f} iou - {metrics[1]:.6f}",
                      f"hassdorf - {metrics[2]:.6f}")
                torch.save(self.model.state_dict(), self.checkpoint_dir + "/" + f'epoch_{epoch + 1}.pth')
        if not test_one:
            self.save_log()

    def plot_metrics(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.losses['train'], label='train')
        plt.plot(self.losses['test'], label='test')
        plt.title('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.dice_scores['train'], label='train')
        plt.plot(self.dice_scores['test'], label='test')
        plt.title('Dice')
        plt.legend()

        plt.subplot(2, 2, 1)
        plt.plot(self.iou_scores['train'], label='train')
        plt.plot(self.iou_scores['test'], label='test')
        plt.title('IOU')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(self.hausdorff_scores['train'], label='train')
        plt.plot(self.hausdorff_scores['test'], label='test')
        plt.title('Hausdorff')
        plt.legend()

        plt.imsave(self.log_dir + "/" + 'plot.png')
        plt.close()

    def save_log(self):
        torch.save(self.model.state_dict(), self.output_dir + "/" + 'last_epoch.pth')

        log = pd.DataFrame({
            'train_loss': self.losses['train'],
            'test_loss': self.losses['test'],
            'train_dice': self.dice_scores['train'],
            'test_dice': self.dice_scores['test'],
            'train_iou': self.iou_scores['train'],
            'test_iou': self.iou_scores['test'],
            'train_hausdorff': self.hausdorff_scores['train'],
            'test_hausdorff': self.hausdorff_scores['test']
        })
        log.to_csv(self.log_dir + "/" + 'log.csv', index=False)

    def load_model(self, path: str):
        print(f"Loading model from {path}")
        self.model.load_state_dict(torch.load(path))


# In[ ]:


def get_paths(path_array):
    if isinstance(path_array, str):
        path_array = [path_array]
    
    allPaths = []
    for path in path_array:
        allPaths.append(sorted(glob(path + '/*/')))
        allPaths.append(sorted(glob(os.path.join(path, '*.nrrd'))))
    return list([item for sublist in allPaths for item in sublist])


# In[ ]:


path_to_image = get_paths(config_dict['dataset']['image_path'])
path_to_masks = get_paths(config_dict['dataset']['mask_path'])
data = [{'image': image, 'mask': mask} for image, mask in zip(path_to_image, path_to_masks)]

train_transforms = Compose([
    LoadImageD(keys=["image", "mask"], reader="itkreader"),
    EnsureChannelFirstd(keys=["image", "mask"]),
    EnsureTyped(keys=["image", "mask"]),
    CenterSpatialCropD(keys=["image", "mask"], roi_size=[400, 400, 200]),
    RandSpatialCropD(keys=["image", "mask"], roi_size=[350, 350, 180], random_size=False),
    ResizeD(keys=["image", "mask"], spatial_size=[256, 256, 112]),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ToTensorD(keys=["image", "mask"]),
])

val_transform = Compose(
    [
        LoadImageD(keys=["image", "mask"], reader="itkreader"),
        EnsureChannelFirstd(keys=["image", "mask"]),
        EnsureTyped(keys=["image", "mask"]),
        CenterSpatialCropD(keys=["image", "mask"], roi_size=[350, 350, 180]),
        ResizeD(keys=["image", "mask"], spatial_size=[256, 256, 112]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensorD(keys=["image", "mask"]),
    ]
)

train_set = monai.data.CacheDataset(data[:int(len(data)*0.8)], transform=train_transforms) if config_dict['dataset']['cache_dataset'] else monai.data.Dataset(data[:int(len(data)*0.8)], transform=train_transforms)
test_set = monai.data.CacheDataset(data[int(len(data)*0.8):], transform=val_transform) if config_dict['dataset']['cache_dataset'] else monai.data.Dataset(data[int(len(data)*0.8):], transform=val_transform)

train_loader = monai.data.DataLoader(train_set, batch_size=config_dict['trainer']['batch_size'], num_workers=config_dict['trainer']['num_workers'], shuffle=True, collate_fn=monai.data.pad_list_data_collate)
test_loader = monai.data.DataLoader(test_set, batch_size=config_dict['trainer']['batch_size'], num_workers=config_dict['trainer']['num_workers'], shuffle=True, collate_fn=monai.data.pad_list_data_collate)

print(f"Train set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")


# In[ ]:


model = UNet(
    spatial_dims=config_dict['model']['spatial_dims'],
    in_channels=config_dict['model']['in_channels'],
    out_channels=config_dict['model']['out_channels'],
    channels=config_dict['model']['channels'],
    strides=config_dict['model']['strides'],
    num_res_units=config_dict['model']['num_res_units'],
)


optimizer = torch.optim.Adam(model.parameters(), config_dict['optimizer']['params']['lr'])


lr_scheduler = ReduceLROnPlateau(optimizer, config_dict['optimizer']['scheduler']['params']['mode'], factor=config_dict['optimizer']['scheduler']['params']['factor'], patience=config_dict['optimizer']['scheduler']['params']['patience'])

criterion = DiceFocalLoss(sigmoid=True, gamma=0.75, squared_pred=True, reduction='mean')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("running on device: ", device)

trainer = Trainer(model, optimizer, criterion, train_loader, test_loader, epochs=config_dict['trainer']['epochs'],
                  lr_scheduler=lr_scheduler, plot=config_dict['trainer']['plot'], device=device,
                  log_dir=config_dict['trainer']['log_dir'], checkpoint_dir=config_dict['trainer']['checkpoint_dir'],
                  output_dir=config_dict['trainer']['output_dir'],
                  checkpoint_interval=config_dict['trainer']['checkpoint_interval'])


# In[ ]:
test_one = False


if len(sys.argv) > 2:
    trainer.load_model(sys.argv[2])
    if sys.argv[1] == 'test':
        test_one = True

if len(sys.argv) == 2:
    if sys.argv[1] == 'test':
        test_one = True

trainer.train()

