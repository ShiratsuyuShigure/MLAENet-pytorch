import os
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from config import Config
from sd import seed_everything

seed_everything()

class CrowdDataset(torch.utils.data.Dataset):


    def __init__(self, root, main_transform=None,use_flip=False):

        main_trans_list=[]
        if use_flip:
            main_trans_list.append(RandomHorizontalFlip())

        img_transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])])
        dmap_transform = ToTensor()


        '''
        root: the root path of dataset.
        phase: train or test.
        main_transform: transforms on both image and density map.
        img_transform: transforms on image.
        dmap_transform: transforms on densitymap.
        '''
        self.img_path = os.path.join(root,'images')
        self.dmap_path = os.path.join(root,'densitymaps')

        self.data_files = [filename for filename in os.listdir(self.img_path)
                           if os.path.isfile(os.path.join(self.img_path, filename))]
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.dmap_transform = dmap_transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        index = index % len(self.data_files)
        fname = self.data_files[index]
        img, dmap = self.read_image_and_dmap(fname)
        if self.main_transform is not None:
            img, dmap = self.main_transform((img, dmap))
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.dmap_transform is not None:
            dmap = self.dmap_transform(dmap)


        return {'image': img, 'densitymap': dmap}

    def read_image_and_dmap(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            print('There is a grayscale image.')
            img = img.convert('RGB')

        dmap = np.load(os.path.join(
            self.dmap_path, os.path.splitext(fname)[0] + '.npy'))
        dmap = dmap.astype(np.float32, copy=False)
        dmap = Image.fromarray(dmap)


        cfg= Config()
        goalx = cfg.x
        goaly = cfg.y

        img = img.resize((goalx, goaly))

        return img, dmap

class RandomHorizontalFlip(object):
    '''
    Random horizontal flip.
    prob = 0.5
    '''

    def __call__(self, img_and_dmap):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap = img_and_dmap
        if random.random() < 0.5:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), dmap.transpose(Image.FLIP_LEFT_RIGHT))
        else:
            return (img, dmap)