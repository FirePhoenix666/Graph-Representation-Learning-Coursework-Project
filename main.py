import os
import numpy as np
import torch

from options import BaseOptions, reset_weight
from trainer import trainer


def set_seed(args):
    # Accelerate our algorithm
    # Details: https://pytorch.org/docs/stable/backends.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Check if GPU is available
    if not torch.cuda.is_available():
        args.cuda = False
        # Sets the seed for generating random numbers
        torch.manual_seed(args.random_seed)
    else:
        args.cuda = True
        # Sets the seed for generating random numbers for the current GPU
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    np.random.seed(args.random_seed)

# Collect arguments
args = BaseOptions().initialize()

# Define seeds used for generating random numbers
seeds = [100, 200, 300, 400, 500]

# Set the number of layers in model
layers = [2, 15, 30]

# Define empty lists to store results of each layer
acc_test_layers = []
MI_XiX_layers = []
dis_ratio_layers = []

# For each number of layers
for layer in layers:
    args.num_layers = layer
    if args.type_norm == 'group':
        args = reset_weight(args)
    # Define empty lists to store results of each seed
    acc_test_seeds = []
    MI_XiX_seeds = []
    dis_ratio_seeds =  []
    # For each seeds
    for seed in seeds:
        args.random_seed = seed
        set_seed(args)
        # Jump to trainer to train models
        trnr = trainer(args)
        # Save results under current seed
        acc_test, MI_XiX, dis_ratio = trnr.train_compute_MI()
        acc_test_seeds.append(acc_test)
        MI_XiX_seeds.append(MI_XiX)
        dis_ratio_seeds.append(dis_ratio)
    # Compute the average results of five seeds
    avg_acc_test = np.mean(acc_test_seeds)
    avg_MI_XiX = np.mean(MI_XiX_seeds)
    avg_dis_ratio = np.mean(dis_ratio_seeds)
    # Save the average results under current number of layers
    acc_test_layers.append(avg_acc_test)
    MI_XiX_layers.append(avg_MI_XiX)
    dis_ratio_layers.append(avg_dis_ratio)

print(f'experiment results of {args.type_norm} applied in {args.type_model} on dataset {args.dataset}')
print('number of layers: ', layers)
print('test accuracies: ', acc_test_layers)
print('instance information gain: ', MI_XiX_layers)
print('group distance ratio: ', dis_ratio_layers)