import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn

import os
import time
from tqdm import tqdm

from utils import create_early_stopper

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


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()


def train(model, loader, val_loader, epochs=100, edge_feature_compact=True):
    """Train and validate the model with given params

    :param edge_feature_compact:
    :param model:
    :param loader:
    :param epochs:
    :return: model
    """
    early_stopper = create_early_stopper(patience=3, min_delta=0.05)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()  # gradscaler for 16-bit calculation

    # Losses
    train_loss_list = []
    val_loss_list = []

    # Accuracies
    train_acc_list = []
    val_acc_list = []

    model.train()
    for epoch in range(epochs + 1):
        total_loss = 0
        acc = 0

        # Train on batches
        for data in loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Mix-precision with autocast
                if edge_feature_compact:
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                else:
                    out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                total_loss += loss / len(loader)
                acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation for each epoch
        val_loss, val_acc = validate_model(model, val_loader, edge_feature_compact)

        val_loss_list.append(val_loss.detach().cpu().numpy().item())
        val_acc_list.append(val_acc)

        train_loss_list.append(total_loss.detach().cpu().numpy().item())
        train_acc_list.append(acc)

        # Print metrics every 20 epochs
        if epoch % 20 == 0:
            print(
                f'Epoch: {epoch:03d}, Train Acc: {acc * 100:.4f}, Val Acc: {val_acc * 100:.4f}, Train Loss: {total_loss:.2f}, Val Loss: {val_loss:.4f}')

        if early_stopper.early_stop(val_loss):
            break

    return model, \
        train_acc_list, \
        train_loss_list, \
        val_acc_list, \
        val_loss_list


@torch.no_grad()
def validate_model(model, loader, edge_feature_compact=True):
    """Evaluate the model on evaluation mode"""
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        with torch.cuda.amp.autocast():  # Mix-precision
            if edge_feature_compact:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)

            loss += criterion(out, data.y) / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    model.train()
    return loss, acc


@torch.no_grad()
def test_model(model, loader, edge_feature_compact=True):
    """Evaluate the model on evaluation mode"""
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        with torch.cuda.amp.autocast():  # Mix-precision
            if edge_feature_compact:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)

            loss += criterion(out, data.y) / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    print(f'Model accuracy: {acc * 100:.3f}')


def visualize_results(train_list, val_list, meta_data):
    """Visualize the loss and valid for train and valid datasets"""
    df_1 = pd.DataFrame({'value': train_list})
    df_1['split'] = 'train'
    df_2 = pd.DataFrame({'value': val_list})
    df_2['split'] = 'valid'
    df = pd.concat([df_1, df_2], axis=0)

    fig = px.line(df, y='value', color='split')
    fig.update_layout(
        # title=meta_data['title'],
        title={
            'text': meta_data['title'],
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title=meta_data['xaxis_title'],
        yaxis_title=meta_data['yaxis_title'],
        legend_title=meta_data['legend'],
        font=dict(
            size=15,
        )
    )
    fig.show()


def run_experiment(model, epochs, edge_feature_compact=True):
    print('Model architecture:\n', model)
    trained_model, \
    train_acc_list, \
    train_loss_list, \
    val_acc_list, \
    val_loss_list = train(model, train_loader, val_loader ,epochs, edge_feature_compact)

    # Visualize the loss
    visualize_results(train_loss_list, val_loss_list, {'title': 'Train and Validation Loss Distribution',
                                                       'xaxis_title': 'Epoch',
                                                       'yaxis_title': 'Loss',
                                                       'legend': 'Split'})
    # Visualize the accuracy
    visualize_results([acc * 100 for acc in train_acc_list],
                      [acc * 100 for acc in val_acc_list],
                      {'title': 'Train and Validation Accuracy Distribution',
                       'xaxis_title': 'Epoch',
                       'yaxis_title': 'Accuracy',
                       'legend': 'Split'})

    return trained_model
