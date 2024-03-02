import os
import timeit
from apex import amp
from sklearn.linear_model import LinearRegression

import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import pylab
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image
from dataset import CrowdDataset
from dataset import create_test_dataloader
from sd import seed_everything
from scipy import signal
from enum import Enum
from model import MLAENet






def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device=torch.device("cuda")
    model=MLAENet()

    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    torch.backends.cudnn.enabled = True
    model= amp.initialize(model,  opt_level="O1")
    dataset=CrowdDataset(img_root,'test')

    #dataloader=torch.utils.data.DataLoader(dataset)
    dataloader =create_test_dataloader(gt_dmap_root)


    model.eval()
    mae=0
    mse=0
    SMAPE=0


    meanG =0.0
    f1=0.0
    f2=0.0
    memE=[]

    x=[]
    y=[]


    time_pool=0

    with torch.no_grad():
        for i,data in enumerate(tqdm(dataloader)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)


            img = data['image'].to(device)
            gt_dmap = data['densitymap'].to(device)
            # forward propagation
            start.record(stream=torch.cuda.current_stream())
            et_dmap=model(img)
            end.record(stream=torch.cuda.current_stream())
            end.synchronize()
            time_pool += start.elapsed_time(end)


            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            mse+=(abs(et_dmap.data.sum()-gt_dmap.data.sum()).item())*(abs(et_dmap.data.sum()-gt_dmap.data.sum()).item())
            SMAPE+=(abs(et_dmap.data.sum()-gt_dmap.data.sum()).item())/((abs(et_dmap.data.sum()+gt_dmap.data.sum()).item()/2))


            EE=et_dmap.data.sum().item()
            GG=gt_dmap.data.sum().item()

            x.append(GG)
            y.append(EE)

            f1 = f1+(EE-GG)*(EE-GG)
            meanG = meanG + GG
            memE.append(EE)



            del img,gt_dmap,et_dmap



    meanG=meanG/len(dataloader)

    SMAPE /= len(dataloader)



    for i in memE:
        f2 = f2 + (i-meanG)*(i-meanG)

    print()
    print("model_param_path:"+model_param_path+" mae:"+str(mae/len(dataloader))+" mse:"+str(mse/len(dataloader)))
    print("R2:"+str(1.0-f1/f2))
    print("SMAPE"+str(SMAPE))


def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cuda")
    model=MLAENet().to(device)
    model.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,'test')
    #dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    dataloader = create_test_dataloader(img_root)
    model.eval()
    for i,data in enumerate(tqdm(dataloader)):
        if i==index:
            img=data['image'].to(device)
            print(img.shape)
            gt_dmap=data['densitymap'].to(device)
            # forward propagation
            et_dmap=model(img).detach()
            print()
            print('pred:')
            print(et_dmap.data.sum().item())
            print('gt:')
            print(gt_dmap.data.sum().item())

            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()

            print(et_dmap.shape)
            toPIL = transforms.ToPILImage()
            img = toPIL(img.squeeze(0))



            #img.show()
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(et_dmap,cmap=CM.jet)
            pylab.show()

            break


if __name__=="__main__":
    seed_everything()
    torch.backends.cudnn.enabled=False
    img_root='data/ShanghaiTech_Crowd_Counting_Dataset/MTC'
    gt_dmap_root='data/ShanghaiTech_Crowd_Counting_Dataset/MTC-UAV'
    model_param_path= 'checkpoints/MLAENet_MTC.pth'

    model=MLAENet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    device = torch.device('cuda')
    model.to(device)


    cal_mae(img_root,gt_dmap_root,model_param_path)


    estimate_density_map(img_root,gt_dmap_root,model_param_path,0)

