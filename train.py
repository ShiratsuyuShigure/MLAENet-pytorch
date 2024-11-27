#%%
import argparse
import math
import random

import numpy as np
import time

import torch
import torch.nn as nn
import os
import sys

from PIL import Image
from tqdm import tqdm

from dataset import CrowdDataset
from sd import seed_everything

from config import Config
from model import MLAENet
from utils import denormalize
from apex import amp
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

seed_everything()




if __name__=="__main__":
    amp.register_float_function(torch, 'sigmoid')

    cfg = Config()                                                          # configuration

    model = MLAENet().to(cfg.device)

    min_mae = sys.maxsize
    min_mae_epoch = -1


    checkpoint_interval = 10


    dataset=CrowdDataset(root=cfg.dataset_root)
    kfold = KFold(n_splits=5, random_state=0, shuffle=True)

    flag = 0
    if flag == 1 :
        path_checkpoint = "checkpoints/checkpoint_10_epoch.pkl"
        checkpoint = torch.load(path_checkpoint)


    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):

        if (flag == 1):
            fold = checkpoint["fold"]
            (train_indices, val_indices) = checkpoint["indices"]

        print("fold "+str(fold+1)+":")
        min_mae = sys.maxsize
        min_mae_epoch = -1


        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,pin_memory=True, num_workers=4,
                                             prefetch_factor=2, persistent_workers=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,pin_memory=True, num_workers=4,
                                             prefetch_factor=2, persistent_workers=True)

        gt_dmap_root = cfg.dataset_root

        model = MLAENet().to(cfg.device)
        criterion = nn.MSELoss(size_average=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        start_epoch = 0
        if (flag == 1):
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            flag=0


        save_f=0


        for epoch in range(start_epoch + 1, cfg.epochs + 1):  # start training

            print("\nepoch " + str(epoch) + ":")
            model.train()
            epoch_loss = 0.0
            for i, data in enumerate(tqdm(train_dataloader)):
                image = data['image'].to(cfg.device)
                gt_densitymap = data['densitymap'].to(cfg.device)
                et_densitymap = model(image)  # forward propagation
                loss = criterion(et_densitymap, gt_densitymap)  # calculate loss
                epoch_loss += loss.item()
                # print("Loss="+str(loss.item()))
                optimizer.zero_grad()
                #loss.requires_grad = True
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()  # back propagation
                optimizer.step()  # update network parameters
            #cfg.writer.add_scalar('Train_Loss', epoch_loss / len(train_dataloader), epoch)
            #lossl_train.append(epoch_loss / len(train_dataloader))

            save_f = save_f + 1


            if epoch % checkpoint_interval == 0:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "fold":fold,
                    "indices":(train_indices, val_indices)
                    }
                path_checkpoint = "./checkpoints/checkpoint_{}_epoch.pkl".format(epoch)
                torch.save(checkpoint, path_checkpoint)

            if epoch >= 50 or epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    epoch_mae = 0.0
                    epoch_loss = 0.0
                    for i, data in enumerate(tqdm(val_dataloader)):
                        image = data['image'].to(cfg.device)
                        gt_densitymap = data['densitymap'].to(cfg.device)
                        et_densitymap = model(image).detach()  # forward propagation
                        loss = criterion(et_densitymap, gt_densitymap)  # calculate loss
                        epoch_loss += loss.item()
                        mae = abs(et_densitymap.data.sum() - gt_densitymap.data.sum())
                        epoch_mae += mae.item()
                    epoch_mae /= len(val_dataloader)

                    if epoch_mae < min_mae:
                        if abs(epoch_mae-min_mae)>0.02:
                            save_f = 0
                        min_mae, min_mae_epoch = epoch_mae, epoch
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.checkpoints,"MSNet" + str(fold+1) +" epoch"+str(epoch)+".pth"))  # save checkpoints

                    if save_f >=50:
                        epoch = max(epoch, 145)
                        save_f = 0

                    print('Epoch ', epoch, ' MAE: ', epoch_mae, ' Min MAE: ', min_mae, ' Min Epoch: ',
                          min_mae_epoch)

# %%

