import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd

class MDN_NN(nn.Module):
    d = 3 # 入力値のサイズ
    t = 1 # 出力値のサイズ（次元数）
    h = 50 # 隠れ層のノード数
    k = 30 # 正規分布の山の数
    
    def __init__(self, d, t, h, k):
        super(MDN_NN, self).__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, k)
        self.fc3 = nn.Linear(h, k)
        self.fc4 = nn.Linear(h, t*k)
        
    def __call__(self, x):
        h       = torch.tanh(self.fc1(x.float()))
        pi      = F.softmax(self.fc2(h),dim=1)
        sigmasq = torch.exp(self.fc3(h))
        mu      = self.fc4(h)
        return pi, sigmasq, mu

    def gaussian_pdf(x, mu, sigmasq):
        return (1 / torch.sqrt(2 * np.pi * sigmasq)) * torch.exp((-1 / (2 * sigmasq)) * torch.norm((x - mu), 2, 1) ** 2)

    def loss(self, pi, sigmasq, mu, target, n):
        losses = Variable(torch.zeros(n))
        for i in range(MDN_NN.k):
            likelihood = MDN_NN.gaussian_pdf(target, mu[:, i*MDN_NN.t:(i+1)*MDN_NN.t], sigmasq[:, i])
            prior = pi[:, i]
            losses += prior * likelihood
        loss = torch.mean(-torch.log(losses))
        return loss
    
def get_df_data(input_path = "plays_2_to_MDN.csv"):
    data_df = pd.read_csv(input_path)
    # play_typeの数値化
    data_df["play_type"] = "nan"
    data_df.loc[data_df["play"]=="guard",  "play_type"] = 0
    data_df.loc[data_df["play"]=="tackle", "play_type"] = 1
    data_df.loc[data_df["play"]=="end",    "play_type"] = 2
    data_df.loc[data_df["play"]=="short",  "play_type"] = 3
    data_df.loc[data_df["play"]=="deep",   "play_type"] = 4
    
    # distanceの数値化
    data_df["dist"] = ""
    data_df.loc[data_df["yardsToGo"]<=3, "dist"] = 3
    data_df.loc[(data_df["yardsToGo"]<=6)&(data_df["yardsToGo"]>=4), "dist"] = 2
    data_df.loc[data_df["yardsToGo"]>=7, "dist"] = 1
    data_df["dist"] = data_df["dist"].astype("float")
    
    # nan除く
    data_df = data_df[data_df["play_type"]!="nan"]
    data_df["play_type"] = data_df["play_type"].astype("float")
    
    return data_df
    
    # 正規化から元に戻す関数
def return_norm(tmp_df, base_df, axis=None):
    """
    tmp_df : 修正するデータ
    base_df : 元にするデータ（min, maxをとるデータ）
    """
    x_max = base_df.max()
    x_min = base_df.min()
    return tmp_df.apply(lambda x: x*(x_max-x_min)+x_min)

def return_norm_(x, base_df, axis=None):
    """
    tmp_df : 修正するデータ
    base_df : 元にするデータ（min, maxをとるデータ）
    """
    x_max = base_df.max()
    x_min = base_df.min()
    return x*(x_max-x_min)+x_min