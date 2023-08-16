import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn

import os
import time
from tqdm import trange


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda param: param.requires_grad, params)

    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)

    if args.opt_sheduler == 'none':
        return None, optimizer
    elif args.opt_sheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)

    return scheduler, optimizer

def train():
    pass

def validate():
    pass

def test():
    pass


def run_experiment():
    pass