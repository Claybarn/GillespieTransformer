# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:30:32 2024

@author: Clayton
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import get_positional_embeddings

class SpikeyDataset(Dataset):
    def __init__(self,data_t,delta_t,data_c,cluster_out_d,time_embed_dim=8,cluster_embed_dim=8,window=15000):
        self.data_t = np.squeeze(data_t.astype(int))
        self.delta_t = np.squeeze(delta_t.astype(int))
        self.data_c = np.squeeze(data_c.astype(int))
        self.window = window
        self.unique_c = np.unique(self.data_c)
        self.cluster_out_d = cluster_out_d
        self.time_embed = get_positional_embeddings(window, cluster_embed_dim)
        self.cluster_embed = get_positional_embeddings(self.cluster_out_d, cluster_embed_dim)
        
    def __len__(self):
        return len(self.data_t)
    def __getitem__(self, idx):
        """ create x """
        start_ind = np.argmax(self.data_t>(self.data_t[idx]-self.window))
        spike_times = self.data_t[start_ind:idx]-self.data_t[idx-1]+self.window-1
        curr_clusters = self.data_c[start_ind:idx]
        curr_time_embed = self.time_embed[spike_times,:]
        curr_cluster_embed = self.cluster_embed[curr_clusters,:]
        x = np.hstack((curr_time_embed,curr_cluster_embed))
        return torch.Tensor(x),(self.delta_t[idx],self.data_c[idx])

class SpikeyDataset2(Dataset):
    def __init__(self,data_t,delta_t,data_c,cluster_out_d,window=1500):
        self.data_t = np.squeeze(data_t.astype(int))
        self.delta_t = np.squeeze(delta_t.astype(int))
        self.data_c = np.squeeze(data_c.astype(int))
        self.unique_c = np.unique(self.data_c)
        self.cluster_out_d = cluster_out_d
        self.window=window
    def __len__(self):
        return len(self.data_t)
    def __getitem__(self, idx):
        """ create x """
        start_ind = np.argmax(self.data_t>(self.data_t[idx]-self.window))
        spike_times = self.data_t[start_ind:idx]-self.data_t[idx-1]+self.window-1
        curr_clusters = self.data_c[start_ind:idx]
        x = np.vstack((spike_times,curr_clusters))
        return torch.tensor(x.T,dtype=torch.long),(self.delta_t[idx],self.data_c[idx])



class SpikeyDatasetAutoregressive(Dataset):
    def __init__(self,data_t,data_c,cluster_out_d,time_embed_dim=8,cluster_embed_dim=8,window=15000):
        self.data_t = np.copy(data_t)
        self.data_c = np.copy(data_c)
        self.window = window
        self.time_embed = get_positional_embeddings(window, time_embed_dim)
        self.cluster_embed = get_positional_embeddings(cluster_out_d, cluster_embed_dim)
    def __len__(self):
        return len(self.data_t)
    def __getitem__(self,idx):
        """ create x """
        start_ind = np.argmax(self.data_t>(self.data_t[-1]-self.window))
        spike_times = self.data_t[start_ind:]-self.data_t[-1]+self.window-1
        curr_clusters = self.data_c[start_ind:].astype(int)
        curr_time_embed = self.time_embed[spike_times,:]
        curr_cluster_embed = self.cluster_embed[curr_clusters,:]
        x = np.hstack((curr_time_embed,curr_cluster_embed))
        return torch.Tensor(x)
    def append(self,t,c):
        self.data_t = np.append(self.data_t,self.data_t[-1]+t)
        self.data_c = np.append(self.data_c,c)


