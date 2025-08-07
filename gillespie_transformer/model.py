# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:27:37 2024

@author: Clayton
"""

import torch
import torch.nn as nn
import copy
from .utils import get_positional_embeddings


class MSA(nn.Module):
    """ Class for Multihead Self Attention """
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class GiTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(GiTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )
    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

class GiT(nn.Module):
  def __init__(self, time_input_d=15000, cluster_input_d=500, n_blocks=2, hidden_d=128, time_embed_d=8,cluster_embed_d=8, n_heads=8, time_out_d=650):
    # Super constructor
    super(GiT, self).__init__()
    # params
    self.time_input_d = time_input_d
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    self.hidden_d=hidden_d
    self.cluster_embed_d = cluster_embed_d
    self.time_embed_d = time_embed_d
    self.time_out_d = time_out_d
    self.cluster_input_d = cluster_input_d
    
    # 1) Linear mapper for tokens
    self.linear_mapper = nn.Linear(self.cluster_embed_d+self.time_embed_d, self.hidden_d)
    
    # 2) Learnable classifiation token. Can reset or add another token if reusing network for new task
    # for neural gillespie, might want a token for time and one for cluster id 
    self.time_token = nn.Parameter(torch.rand(1, self.hidden_d))
    self.cluster_token = nn.Parameter(torch.rand(1, self.hidden_d))
    
    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([GiTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
    
    # 5) Classification MLPk
    self.time_mlp = nn.Sequential(
        nn.Linear(self.hidden_d, time_out_d),
        nn.Softmax(dim=-1)
    )
    self.cluster_mlp = nn.Sequential(
        nn.Linear(self.hidden_d, cluster_input_d),
        nn.Softmax(dim=-1)
    )
    
  def forward(self, x):
    """ x is spike times concat with embedding of cluster id (batches ,num tokens, times + cluster id embedding) """
    # 1)
    tokens = self.linear_mapper(x)
    # 2)
    out = torch.stack([torch.vstack((self.time_token,self.cluster_token, tokens[i])) for i in range(len(tokens))])
    # 3) # can skip positional embedding since it comes for free with this data
    # 4)
    for block in self.blocks:
        out = block(out)
    # Getting the classification tokens only
    time_out_token = out[:, 0]
    cluster_out_token = out[:, 1]
    # stack time and cluster id distributions
    out = torch.hstack([self.time_mlp(time_out_token),self.cluster_mlp(cluster_out_token)])
    return out








class GiT3(nn.Module):
  def __init__(self, time_input_d=1500, cluster_input_d=500, n_blocks=2, hidden_d=64, n_heads=8, time_out_d=650):
    # Super constructor
    super(GiT3, self).__init__()
    # params
    self.time_input_d = time_input_d
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    self.hidden_d=hidden_d
    self.time_out_d = time_out_d
    self.cluster_input_d = cluster_input_d
    self.time_embed = nn.Parameter(torch.rand(self.time_input_d+1, self.hidden_d)) # +1 for padded token
    self.cluster_embed = nn.Parameter(torch.rand(self.cluster_input_d+1, self.hidden_d)) # +1 for padded token
    # 2) Learnable classifiation token. Can reset or add another token if reusing network for new task
    # for neural gillespie, might want a token for time and one for cluster id 
    self.time_token = nn.Parameter(torch.rand(1, self.hidden_d*2))
    self.cluster_token = nn.Parameter(torch.rand(1, self.hidden_d*2))
    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([GiTBlock(hidden_d*2, n_heads) for _ in range(n_blocks)])
    # 5) Classification MLPk
    self.time_mlp = nn.Sequential(
        nn.Linear(self.hidden_d*2, time_out_d)
    )
    self.cluster_mlp = nn.Sequential(
        nn.Linear(self.hidden_d*2, cluster_input_d)
    )
  def forward(self, x):
    """ x is spike times concat with embedding of cluster id (batches ,num tokens, times + cluster id embedding) """
    time_inds = x[:,:,0]+1 # time indices, +1 to make zero map to masked token (-1 padded to make length the same)
    time_embed = self.time_embed[time_inds[:],:].reshape(time_inds.size(0),time_inds.size(1),self.hidden_d)
    cluster_inds = x[:,:,1]+1 # cluster indices, +1 to make zero map to masked token  (-1 padded to make length the same)
    cluster_embed = self.cluster_embed[cluster_inds[:],:].reshape(cluster_inds.size(0),cluster_inds.size(1),self.hidden_d)
    # x is spike times concat with embedding of cluster id (batches ,num tokens, times + cluster id embedding) 
    tokens = torch.cat((time_embed,cluster_embed),dim=2)
    # 2)
    out = torch.stack([torch.vstack((self.time_token, self.cluster_token, tokens[i])) for i in range(len(tokens))])
    # 4)
    for block in self.blocks:
        out = block(out)
    # Getting the classification tokens only
    time_out_token = out[:, 0]
    cluster_out_token = out[:, 1]
    # stack time and cluster id distributions
    out = torch.hstack([self.time_mlp(time_out_token),self.cluster_mlp(cluster_out_token)])
    return out





















class GiT2(nn.Module):
  def __init__(self, time_input_d=15000, cluster_input_d=500, time_out_d=650, time_embed_d=2, cluster_embed_d=16, hidden_d=128, n_blocks=2, n_heads=8):
    # Super constructor
    super(GiT2, self).__init__()
    
    # input params
    self.time_input_d = time_input_d
    self.cluster_input_d = cluster_input_d
    
    # output params
    self.time_out_d = time_out_d
    
    # embedding params
    self.time_embed_d = time_embed_d
    self.cluster_embed_d = cluster_embed_d
    self.hidden_d = hidden_d
    
    # transformer params
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    
    # 0) Embeddings (unlearnable and learnable, respectively)
    self.time_embed = get_positional_embeddings(self.time_input_d, self.time_embed_d)
    self.cluster_embed = nn.Parameter(torch.rand(self.cluster_input_d, self.cluster_embed_d))
    
    # 1) Linear mapper for tokens
    self.linear_mapper = nn.Linear(self.cluster_embed_d+self.time_embed_d, self.hidden_d)
    
    # 2) Learnable classifiation token. Can reset or add another token if reusing network for new task
    self.time_token = nn.Parameter(torch.rand(1, self.hidden_d))
    self.cluster_token = nn.Parameter(torch.rand(1, self.hidden_d))
    
    # 3) Transformer encoder blocks
    self.blocks = nn.ModuleList([GiTBlock(hidden_d,self. n_heads) for _ in range(self.n_blocks)])
    
    # 4) Classification MLPk
    self.time_mlp = nn.Sequential(
        nn.Linear(self.hidden_d, time_out_d),
        nn.Softmax(dim=-1)
    )
    self.cluster_mlp = nn.Sequential(
        nn.Linear(self.hidden_d, cluster_input_d),
        nn.Softmax(dim=-1)
    )
    
  def forward(self, x):
    time_inds = x[:,:,0] # time indices
    time_embed = self.time_embed[time_inds[:],:].reshape(time_inds.size(0),time_inds.size(1),self.time_embed_d)
   
    cluster_inds = x[:,:,1] # cluster indices
    cluster_embed = self.cluster_embed[cluster_inds[:],:].reshape(cluster_inds.size(0),cluster_inds.size(1),self.cluster_embed_d)
    
    # x is spike times concat with embedding of cluster id (batches ,num tokens, times + cluster id embedding) 
    x = torch.cat((time_embed,cluster_embed),dim=2)
    
    # 1)
    tokens = self.linear_mapper(x)
    # 2)
    out = torch.stack([torch.vstack((self.time_token,self.cluster_token, tokens[i])) for i in range(len(tokens))])
    # 3) # can skip positional embedding since it comes for free with this data
    # 4)
    for block in self.blocks:
        out = block(out)
    # Getting the classification tokens only
    time_out_token = out[:, 0]
    cluster_out_token = out[:, 1]
    # stack time and cluster id distributions
    out = torch.hstack([self.time_mlp(time_out_token),self.cluster_mlp(cluster_out_token)])
    return out


class MSAAttention(nn.Module):
    """ Class for Multihead Self Attention that returns attention """
    def __init__(self, d, n_heads=2):
        super(MSAAttention, self).__init__()
        self.d = d
        self.n_heads = n_heads
        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k = q_mapping(seq), k_mapping(seq)
                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class GiTBlockAttention(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(GiTBlockAttention, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSAAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )
    def forward(self, x):
        return self.mhsa(self.norm1(x))


class AttentionSiphon(nn.Module):
    """
    Sucks the attention out of our model (like a bat)
    /|\ ^._.^ /|\ 
    """
    def __init__(self, model):
        # Super constructor
        super(AttentionSiphon, self).__init__()
        self.model = copy.deepcopy(model)
        self.model.blocks[-1] = GiTBlockAttention(model.hidden_d, model.n_heads)
    def forward(self,x):
        tokens = self.model.linear_mapper(x)
        out = torch.stack([torch.vstack((self.model.time_token,self.model.cluster_token, tokens[i])) for i in range(len(tokens))])
        for block in self.model.blocks:
            out = block(out)
        return self.extract_attention(out)
    def extract_attention(self,x):
        """ x should be output of final block. Assumes batch size 1! """
        x = x.detach().reshape(x.size(0),x.size(1),self.model.n_heads,x.size(1)) # (batch size, num tokens, num heads, num tokens)
        x = torch.mean(x,2) 
        timing_attn = x[:,0,:]
        id_attn = x[:,1,:]
        return {'timing': timing_attn,'id': id_attn }

