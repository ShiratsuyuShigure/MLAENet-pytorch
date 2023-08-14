# MLAENet-pytorch
This is an implemention of MLAENet

The project will take some time to sort out......


## Data Setup
1. Download MTC and MTC UAV Dataset from

  https://github.com/poppinace/mtc

  https://github.com/poppinace/mtc-uav

2. Put MTC Dataset in 'ROOT/data'. 
3. You can find two python scripts in 
'data_preparation' folder which are used to generate ground truth density-map for 
MTC and MTC-UAV respectively. (Mind that you need move the script to corresponding sub dataset folder like 'ROOT/data/MTC' and run it)  
## Train
1. Modify the dataset root in 'config.py'   
2. Run 'train.py'

## Testing
1. Run 'test.py' for calculate MAE of test images or just show an estimated density-map. 

You can download the pretrained model from https://drive.google.com/file/d/1nCdEb16gLsjcmrBgX44CFitaq0SN2I4S/view?usp=drive_link
