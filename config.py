import torch
import os


class Config():
    '''
    Config class
    '''
    def __init__(self):
        self.dataset_root = 'data/MTC'
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.lr = 4e-6              # learning rate

        if self.lr == 4e-6:
            self.x = 1024
            self.y = 768
        else:
            self.x = 1368
            self.y = 912

        self.batch_size   = 4                 # batch size
        self.epochs       = 300               # epochs
        self.checkpoints  = './checkpoints'     # checkpoints dir
        #self.writer       = SummaryWriter()     # tensorboard writer

        self.__mkdir(self.checkpoints)

    def __mkdir(self, path):
        '''
        create directory while not exist
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            print('create dir: ',path)
