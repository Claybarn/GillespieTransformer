# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:49:17 2024

@author: Clayton
"""

import os
import builtins
import argparse
import torch
import numpy as np 
import random
import torch.distributed as dist
import torch.utils.data as data
from torch.nn import CrossEntropyLoss
from torch.distributions import Categorical
from gillespie_transformer import GiT, ClusterMapping, SpikeyDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spike_times_file', default='spike_times.npy', type=str)
    parser.add_argument('--spike_clusters_file', default='spike_clusters.npy', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size per GPU')
    parser.add_argument('--batches_per_step', default=128, type=int, help='batches per weight update')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers for loading data')
    args = parser.parse_args()
    return args
                                         
def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    #ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    """ data """
    data_t=np.squeeze(np.load(args.spike_times_file))
    data_c=np.squeeze(np.load(args.spike_clusters_file))
    delta_t=np.zeros_like(data_t)
    delta_t[1:]=np.diff(data_t)
    cluster_mapping = ClusterMapping(data_c)
    
    cluster_out_d = cluster_mapping.num_clusters
    time_out_d = np.max(delta_t)+1

    data_c = cluster_mapping.forward_map(data_c)
    
    cutoff = int(len(data_t)*.8)
    train_t = data_t[:cutoff]
    train_dt = delta_t[:cutoff]
    train_c = data_c[:cutoff]
    
    test_t = data_t[cutoff:]
    test_dt = delta_t[cutoff:]
    test_c = data_c[cutoff:]
    
    train_dataset = SpikeyDataset(train_t,train_dt,train_c,cluster_out_d)
    val_dataset = SpikeyDataset(test_t,test_dt,test_c,cluster_out_d)
    
    train_sampler = data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
            collate_fn=collate_fn)
    
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True,
            collate_fn=collate_fn)
    
    
    ### model ###
    model = GiT(cluster_out_d=cluster_out_d,time_out_d=time_out_d)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            #model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            #model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    """ optimizer """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    """ calculate class weights """
    unique_times,unique_counts = np.unique(delta_t,return_counts=True)
    temporal_weights = np.ones(time_out_d)*-np.inf
    temporal_weights[unique_times] = unique_counts
    temporal_weights = np.max(temporal_weights) / temporal_weights
    
    unique_clusters,unique_counts = np.unique(data_c,return_counts=True)
    id_weights = np.max(unique_counts)/unique_counts
    
    """ define criterion for validation """
    criterion = CrossEntropyLoss()
    
    """ TODO: implement resume traning """
    if args.resume:
        pass
    
    """ define training vars """
    best_test_loss = np.inf
    
    torch.backends.cudnn.benchmark = True
    
    """main loop""" 
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)
        
        # adjust lr if needed #
        
        train_one_epoch(train_loader, model, temporal_weights, id_weights, optimizer, epoch, args)
        if args.rank == 0: # only val and save on master node
            test_loss = validate(val_loader, model, criterion, epoch, args)
            # save checkpoint if needed #
            if test_loss < best_test_loss:
                print('saving...')
                best_test_loss = test_loss
                torch.save(
                    {"model": model.state_dict()},
                    'GiT.model',
                )


def train_one_epoch(train_loader, model, temporal_weights, id_weights, optimizer, epoch, args):
    """ 
    only one gpu is visible here, so you can send cpu data to gpu by 
    input_data = input_data.cuda() as normal
    """
    train_loss = 0.0
    # define 
    temporal_criterion = CrossEntropyLoss(weight=torch.tensor(temporal_weights).cuda().float())
    id_criterion = CrossEntropyLoss(weight=torch.tensor(id_weights).cuda().float())
    for it,batch in enumerate( train_loader):
        x, y = batch
        y_hat = model(x.cuda())
        t_loss = temporal_criterion(y_hat[:,:model.time_out_d], y[:,0].cuda())
        c_loss = id_criterion(y_hat[:,model.time_out_d:], y[:,1].cuda())
        # exponential decay on confidence penalty
        loss =  t_loss + c_loss - 0.1*np.exp(-epoch)*Categorical(probs = y_hat[:,:model.time_out_d]).entropy() -  0.1*np.exp(-epoch)*Categorical(probs = y_hat[:,model.time_out_d:]).entropy()
        train_loss += loss.detach().cpu().item()/len(train_loader)
        loss = torch.mean(loss)
        loss.backward()
        if (it+1)>=args.batches_per_step == 0 :
            optimizer.step()
            optimizer.zero_grad()
    print("Epoch " + str(epoch) + ": " + str(train_loss))


def validate(val_loader, model, criterion, optimizer, epoch, args):
    with torch.no_grad():
        test_loss = 0.0
        for batch in val_loader:
            x, y = batch
            y_hat = model(x.cuda())
            loss = criterion(y_hat[:,:model.time_out_d], y[:,0].cuda())  + criterion(y_hat[:,model.time_out_d:], y[:,1])
            loss = torch.mean(loss)
            test_loss += loss.detach().cpu().item() / len(val_loader)
        print(test_loss)
        return test_loss


if __name__ == '__main__':
    args = parse_args()
    main(args)