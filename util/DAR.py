
import torch
from torch.autograd.variable import Variable
from torch.optim import Adam, SGD
import lightning.pytorch as pl
from torchmetrics import MeanMetric,MinMetric

import numpy as np
from functools import partial
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset,IterableDataset,Dataset
import lightning.pytorch as pl
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import LearningRateFinder,LearningRateMonitor
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from collections import Counter
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import joblib
# from MODELS import cbam
import math
import torch.nn as nn
import yaml
from pytorch_lightning.loggers import TensorBoardLogger


class DAR_data(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str="./dataset",
            x_data_s_path:str="x_data.csv",
            y_data_s_path:str="y_data.csv",
            x_data_t_path:str="x_data_t.csv",
            discriptor: list = ["a"],
            random_seed: int = 42,
            batch: int = 32,
            normal_y:bool = False,
            normal_x:bool = False,
            Yscaling_method: str = "minmax",
            Xscaling_method: str = "minmax",
    ):
        super().__init__()
        self.data_dir = data_path
        self.x_data_s_path = x_data_s_path
        self.y_data_s_path = y_data_s_path
        self.x_data_t_path = x_data_t_path
        self.random_seed = random_seed
        self.batch = batch
        self.discriptor = discriptor
        self.normal_y = normal_y
        self.normal_x = normal_x
        self.Yscaling_method = Yscaling_method
        self.Xscaling_method = Xscaling_method
    def prepare_data_train(self):
        x_data_s = pd.read_csv(self.data_dir + "/" + self.x_data_s_path)
        x_data_s=x_data_s.drop(columns=['E'])
        y_data_s = pd.read_csv(self.data_dir + "/" + self.y_data_s_path)
        x_data_t = pd.read_csv(self.data_dir + "/" + self.x_data_t_path)
        x_data_t = x_data_t.drop(columns=['E'])
        if self.normal_x:
            x_data_s = self.scaler(x_data_s, self.Xscaling_method)
            x_data_t = self.scaler(x_data_t, self.Xscaling_method)
        if self.normal_y:
            y_data_s = self.scaler(y_data_s, self.Yscaling_method)
        data=[]
        keys=list(x_data_s.keys())
        keys_label=list(y_data_s.keys())    
        for i in range(len(keys)):
            data.append([np.array(x_data_s[keys[i]]),np.array(y_data_s[keys_label[i]]),np.array(x_data_t[keys[i]])])
        # print(len(data))
        return data
    def prepare_data_test(self,
                          data_test_path:str,label_test_path:str):
        x_data_s = pd.read_csv(data_test_path)
        x_data_s=x_data_s.drop(columns=['E'])
        data=[]
        keys=list(x_data_s.keys())
        for i in range(len(keys)):
            data.append(x_data_s[keys[i]])
        return data
    def setup(self,
        stage: str,
        batch: int = 0,
        x_data_path: str = "",
        label_data_path: str = "",
        Emesh_test=None
        ):
        if stage == "fit":
            sample = self.prepare_data_train()
            self.train_data_func = partial(data_generator,sample, True)
            self.val_data_func = partial(data_generator, sample, False)
        elif stage == "test":
            sample = self.prepare_data_train()
            self.test_data_func = partial(data_generator, sample, False) 
        elif stage == "test_other":
            sample = self.prepare_predict_data(
                x_data_path, label_data_path, Emesh_test,stage='test'
            )
            
            self.test_other_data_func = partial(data_test_other_generator, sample) 
        elif stage == "predict":
            sample = self.prepare_predict_data(
                x_data_path, label_data_path,stage='predict'
            )
            return sample
    def train_dataloader(self):
        return GeneratorDataset(self.train_data_func, self.batch)

    def val_dataloader(self):
        return GeneratorDataset(self.val_data_func, self.batch)

    def test_other_dataloader(self, batch: int = 10):
        return GeneratorDataset(self.test_other_data_func, batch_size=batch)

def data_generator(data, train:bool, batch_size):
    if batch_size > len(data):
        raise ValueError("batch_size should be smaller than the length of data")

    while True:
        if train == True:
            x_batch_s,x_batch_t,y_batch = generate_examples(batch_size, data)
        else:
            ran_batch = np.random.randint(0, len(data), batch_size)
            x_batch_s = []
            x_batch_t = []
            y_batch = []
            for index in ran_batch:
                x_batch_s.append(data[index][0])
                x_batch_t.append(data[index][2])
                y_batch.append(data[index][1])
        x_batch_s = np.asarray(x_batch_s)
        x_batch_t = np.asarray(x_batch_t)
        y_batch = np.asarray(y_batch)
        x_batch_s = x_batch_s.reshape(x_batch_s.shape[0], x_batch_s.shape[1])
        x_batch_t = x_batch_t.reshape(x_batch_t.shape[0], x_batch_t.shape[1])
        y_batch = y_batch.reshape(y_batch.shape[0], y_batch.shape[1])
        yield torch.from_numpy(x_batch_s.astype(np.float32)), torch.from_numpy(x_batch_t.astype(np.float32)),torch.from_numpy(
            y_batch.astype(np.float32))
        
def generate_examples(num_examples, data):
    """
    data contains spectra and labels
    """
    x_examples_s = []
    x_examples_t=[]
    label_examples_s = []
    num_data = len(data)
    ran_com = np.random.randint(0,num_data,num_examples)
    for i in ran_com:
        x_examples_s.append(data[i][0])
        x_examples_t.append(data[i][2])
        label_examples_s.append(data[i][1])
    x_s_examples = np.asarray(x_examples_s)
    x_t_examples = np.asarray(x_examples_t)
    label_examples_s = np.asarray(label_examples_s)
    return x_s_examples, x_t_examples, label_examples_s



class Vani_NN(nn.Module):
    def __init__(self,feature_len=200,label_len=84,config_para=dict()):
        super(Vani_NN, self).__init__()
        self.feature_len=feature_len
        self.label_len=label_len
        self.config_para=config_para
        # kernel size must be odd
        self.NN=torch.nn.Sequential(
            nn.Linear(feature_len, self.config_para['linear1']),
            nn.ReLU(),
            nn.Linear(self.config_para['linear1'],self.config_para['linear2']),
            nn.ReLU(),
            nn.Linear(self.config_para['linear2'],self.label_len))
    def forward(self,x):
        return self.NN(x)
    


class NN_model(pl.LightningModule):
    def __init__(self,net:torch.nn.Module,loss_func,tradeoff,tradeoff2,lr=1e-1):
        super().__init__()
        self.net=net
        self.loss=loss_func
        self.train_loss=MeanMetric()
        self.val_loss=MeanMetric()
        self.tradeoff=tradeoff
        self.tradeoff2=tradeoff2
        self.lr=lr
    def forward(self,x):
        return self.net(x)
    def training_step(self,batch,batch_idx):
        x_s,x_t,y=batch
        y_hat_s=self.net(x_s)
        y_hat_t=self.net(x_t)
        loss_reg=self.loss(y_hat_s,y)
        loss_reg+=self.tradeoff*RSD(y_hat_s,y_hat_t,self.tradeoff2)
        # print(y_hat_s,y_hat_t)
        loss=self.train_loss(loss)
        # print(loss)
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self,batch,batch_idx):
        x_s,x_t,y=batch
        y_hat_s=self.net(x_s)
        loss_reg=self.loss(y_hat_s,y)
        loss=loss_reg
        loss=self.val_loss(loss)
        self.log('val_loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(),lr=self.lr)
        return optimizer
    
    
class GeneratorDataset(IterableDataset):
    def __init__(self, gen_func,batch_size):
        super().__init__()
        self.gen_func = gen_func(batch_size)

    def __iter__(self):
        return self.gen_func

class GeneratorDataLoader(DataLoader):
    def __init__(self, gen_func, batch_size=1,shuffle=False, num_workers=0):
        dataset = GeneratorDataset(gen_func,batch_size)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def RSD(Feature_s, Feature_t,tradeoff2):
    u_s, s_s, v_s = torch.svd(Feature_s.t())
    u_t, s_t, v_t = torch.svd(Feature_t.t())
    p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    return torch.norm(sinpa,1)+tradeoff2*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)