# Gillespie Transformer (GiT)

A neural transformer model for predicting neural spike timing and neuron identification, inspired by the Gillespie algorithm for stochastic event simulation.

## Overview

The Gillespie Transformer (GiT) is a specialized transformer architecture designed to predict:
1. The time until the next neural spike
2. Which neuron will fire next

This model combines the powerful attention mechanisms of transformers with the principles of the Gillespie algorithm, making it particularly well-suited for modeling stochastic neural firing patterns.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Distributed training support (NCCL backend)

## Installation

```bash
git clone https://github.com/yourusername/GillespieTransformer.git
cd GillespieTransformer
# Install required packages
pip install torch numpy
```

## Data Format

The model expects two main input files:
- `spike_times.npy`: Numpy array containing spike timing information
- `spike_clusters.npy`: Numpy array containing neuron cluster/ID information

## Usage

### Training

The model supports distributed training using PyTorch's DistributedDataParallel. To train the model:

```bash
python main.py \
    --spike_times_file spike_times.npy \
    --spike_clusters_file spike_clusters.npy \
    --window 15000 \
    --batch_size 32 \
    --epochs 10 \
    --lr 1e-3
```

### Key Parameters

- `--window`: Maximum window size for looking into past spikes (default: 15000)
- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Batch size per GPU (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--workers`: Number of data loading workers (default: 1)

### Distributed Training

The model is designed to work with distributed training. For multi-GPU training:

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main.py
```

## Model Architecture

The Gillespie Transformer (GiT) consists of:
- A transformer-based architecture for processing temporal sequences
- Dual output heads for predicting:
  - Time until next spike
  - Next neuron ID to fire
- Custom attention mechanisms optimized for neural spike prediction

## Data Structure

The model processes neural spike data with:
- Temporal information (spike timing)
- Spatial information (neuron IDs/clusters)
- Automatically calculated inter-spike intervals

## Output

The model produces two predictions:
1. Time prediction: When the next spike will occur
2. Neuron prediction: Which neuron will generate the next spike

## Checkpointing

The model automatically saves checkpoints during training, with files named as `GiT{epoch}.model`.

