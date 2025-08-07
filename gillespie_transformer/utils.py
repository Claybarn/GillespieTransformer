# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:07:18 2024

@author: Clayton
"""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class ClusterMapping():
    def __init__(self,clusters):
        clusters = clusters.astype(int)
        unique_clusters = np.unique(clusters)
        self.num_clusters = len(unique_clusters)
        
        """ build map to convert cluster ids to continuous cluster ids """
        self.mapping = np.nan*np.empty(unique_clusters[-1]+1,dtype = int)
        for i,c in enumerate(unique_clusters):
            self.mapping[c] = i
        
        """ build map to convert continuous cluster ids to original cluster ids """
        self.inverse_mapping = np.nan*np.empty(self.num_clusters,dtype = int)
        for i,c in enumerate(unique_clusters):
            self.inverse_mapping[i] = c
        
    def forward_map(self,clusters):
        """ map cluster ids to continuous cluster ids """
        return self.mapping[clusters.astype(int)]
    def inverse_map(self,clusters):
        """ map continuous cluster ids to original cluster ids """
        return self.inverse_mapping[clusters.astype(int)]


def collate_fn(batch):
    data, targets = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True,padding_value=-1)    
    targets = torch.tensor(targets,dtype=torch.long)
    return [data_padded, targets]