
# 1. Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

import config_param
from model.unet_model_architecture import UNetModel

import dataset.data_loaders_fold_blockcross_subsampling as utils
from dataset.calperum_dataset import CalperumDataset
##change to this when main.py or main_subsampling.py
# from dataset.calperum_dataset_main import CalperumDataset

# # PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from dataset.data_augmentation import get_transform 


############# 3_TRAINING MODULE ##############
'''Using PyTorch Lightning simplifies the training loop, validation, and testing procedures.
'''
# 4. PyTorch Lightning Module
class UNetModule(pl.LightningModule):
    def __init__(self):
        super(UNetModule, self).__init__()
        self.model = UNetModel(
            in_channels=config_param.IN_CHANNELS,
            out_channels=config_param.OUT_CHANNELS
        )
        self.learning_rate = config_param.LEARNING_RATE
        self.criterion = config_param.CRITERION
        # self.criterion = criterion
        # self.criterion = config_param.CRITERION(weights=class_weights, device=config_param.DEVICE)
        self.batch_size = config_param.BATCH_SIZE
        self.num_workers = config_param.NUM_WORKERS
        self.image_folder = config_param.IMAGE_FOLDER
        self.mask_folder = config_param.MASK_FOLDER
        self.transform = config_param.DATA_TRANSFORM
        
        # Initialize metrics for detailed evaluation
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=config_param.OUT_CHANNELS, ignore_index=-1)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=config_param.OUT_CHANNELS, average='none', ignore_index=-1)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=config_param.OUT_CHANNELS, average='none', ignore_index=-1)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=config_param.OUT_CHANNELS, ignore_index=-1, average='none')
        self.iou = JaccardIndex(task="multiclass", num_classes=config_param.OUT_CHANNELS, ignore_index=-1, average='none')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, masks.squeeze(1).long())
        
        # Log training metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.accuracy(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('train_precision', self.precision(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('train_recall', self.recall(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('train_f1', self.f1_score(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('train_iou', self.iou(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, masks.squeeze(1).long())
        
        # Log validation metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.accuracy(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('val_precision', self.precision(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('val_recall', self.recall(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('val_f1', self.f1_score(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('val_iou', self.iou(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, masks.squeeze(1).long())
        
        # Log test metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', self.accuracy(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('test_precision', self.precision(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('test_recall', self.recall(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('test_f1', self.f1_score(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        self.log('test_iou', self.iou(preds, masks.squeeze(1)), on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return config_param.OPTIMIZER(self.parameters(), lr=config_param.LEARNING_RATE)

    def train_dataloader(self):
        return utils.get_train_loader(self.image_folder, self.mask_folder, self.transform)

    def val_dataloader(self):
        return utils.get_val_loader(self.image_folder, self.mask_folder, self.transform)

    def test_dataloader(self):
        return utils.get_test_loader(self.image_folder, self.mask_folder, self.transform)
    # def train_dataloader(self):
    #     train_transform = get_transform(train=True, enable_augmentation=config_param.APPLY_TRANSFORMS)
    #     train_dataset = CalperumDataset(
    #         image_folders=config_param.IMAGE_FOLDER,
    #         mask_folders=config_param.MASK_FOLDER,
    #         transform=train_transform
    #     )
    #     train_loader = DataLoader(train_dataset, batch_size=config_param.BATCH_SIZE, shuffle=True, num_workers=config_param.NUM_WORKERS)
        
    #     # Debugging: Inspect a batch of augmented data
    #     for batch in train_loader:
    #         images, masks = batch
    #         print("Augmented Images Shape:", images.shape)
    #         print("Augmented Masks Shape:", masks.shape)
    #         break  # Inspect only the first batch

    #     return train_loader

    # def val_dataloader(self):
    #     val_transform = get_transform(train=False, enable_augmentation=False)
    #     val_dataset = CalperumDataset(
    #         image_folders=config_param.IMAGE_FOLDER,
    #         mask_folders=config_param.MASK_FOLDER,
    #         transform=val_transform
    #     )
    #     return DataLoader(val_dataset, batch_size=config_param.BATCH_SIZE, shuffle=False, num_workers=config_param.NUM_WORKERS)

    # def test_dataloader(self):
    #     test_transform = get_transform(train=False, enable_augmentation=False)
    #     test_dataset = CalperumDataset(
    #         image_folders=config_param.IMAGE_FOLDER,
    #         mask_folders=config_param.MASK_FOLDER,
    #         transform=test_transform
    #     )
    #     return DataLoader(test_dataset, batch_size=config_param.BATCH_SIZE, shuffle=False, num_workers=config_param.NUM_WORKERS)