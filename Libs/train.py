################### - Imports - #############################################################
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
import pandas as pd


from sklearn.metrics import auc

from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import BinaryAveragePrecision
from torchmetrics.classification import BinaryPrecisionRecallCurve

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn

import os
import time
import shutil
import yaml
import json
from tqdm import tqdm

from Libs.utils import create_early_stopper, seed_everything
from Libs.logger import get_logger

import warnings
warnings.filterwarnings("ignore")
##################################
############################################################################################


################### - Paths - #############################################################
base_path_for_saving_artifacts = './Runs'


##################################
############################################################################################


################### - Modular Functions - #############################################################
def _create_directories(experiment_name):
    """Creating directories for saving artifacts for a specific experiment / run"""

    # For the base path
    if not os.path.exists(base_path_for_saving_artifacts):
        os.makedirs(base_path_for_saving_artifacts, exist_ok=True)

    path_to_saving = os.path.join(
        base_path_for_saving_artifacts, str(experiment_name))

    # Remove the old experiment
    if os.path.exists(path_to_saving):
        shutil.rmtree(path_to_saving)

    # For saving images
    os.makedirs(os.path.join(path_to_saving, 'images'), exist_ok=True)
    # For saving model weights
    os.makedirs(os.path.join(path_to_saving, 'models'), exist_ok=True)

    return path_to_saving


def build_optimizer(args, params):
    """Build an optimizer and scheduler"""

    weight_decay = args.weight_decay
    filter_fn = filter(lambda param: param.requires_grad, params)

    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr,
                               weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr,
                              momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(filter_fn, lr=args.lr,
                               weight_decay=weight_decay)

    if args.opt_sheduler == 'none':
        return None, optimizer
    elif args.opt_sheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart)
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)

    return scheduler, optimizer


def l2_norm(model):
    """Calculate the l2 norm of the given model parameters,
       Lower the result, better it is...
    """

    # Initialize the norm to zero
    norm = 0

    # Loop through all the parameters in the model
    for param in model.parameters():
        # Add the l2 norm of the parameter to the total norm
        norm += torch.sum(param**2)

    # get the square root to get the overall L2 norm
    norm = torch.sqrt(norm)

    return norm 


def accuracy(pred_y, y):
    """Accuracy of the model: supervised learning"""
    pred_y = torch.sigmoid(pred_y)
    return ((pred_y.round() == y).sum() / len(y)).item()


def compute_roc_auc_score(pred_logits, y):
    
    # Taking the probs
    pred_y = torch.sigmoid(pred_logits)

    metric = BinaryAUROC(thresholds=None)

    return metric(pred_y, y)


def compute_avg_precision_score(pred_logits, y):

    pred_y = torch.sigmoid(pred_logits)

    metric = BinaryAveragePrecision(thresholds=None)

    return metric(pred_y, y)


def compute_PR_curve(pred_logits, y):

    pred_y = torch.sigmoid(pred_logits)

    metric = BinaryPrecisionRecallCurve(thresholds=None)

    return metric(pred_y, y) 


def train(model, loader, val_loader, epochs, edge_feature_compact, path_to_saving_artifacts, enable_early_stopping, logger, device, criterion):
    """Train and validate the model with given params

    :param val_loader:
    :param edge_feature_compact:
    :param model:
    :param loader:
    :param epochs:
    :return: model
    """
    model.to(device)

    early_stopper = create_early_stopper(patience=10, min_delta=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01)
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

        # Decay Learning Rate
        scheduler.step()
        # print('Epoch:', epoch,'LR:', scheduler.get_lr())

        # Train on batches
        for data in loader:
            data.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Mix-precision with autocast
                if edge_feature_compact:
                    out = model(data.x.to(torch.float), data.edge_index,
                                data.edge_attr.to(torch.float), data.batch)
                else:
                    out = model(data.x.to(torch.float), data.edge_index, data.batch)
                    
                loss = criterion(out, data.y.flatten().float())
                total_loss += loss / len(loader)

                acc += accuracy(out, data.y.flatten().to(torch.int64)) / len(loader)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation for each epoch
        val_loss, val_acc = validate_model(model, val_loader, device, criterion, edge_feature_compact)

        val_loss_list.append(val_loss.detach().cpu().numpy().item())
        val_acc_list.append(val_acc)

        train_loss_list.append(total_loss.detach().cpu().numpy().item())
        train_acc_list.append(acc)

        # Print metrics every 20 epochs
        if epoch % 20 == 0:
            print(
                f'Epoch: {epoch:03d}, Train Acc: {acc * 100:.4f}, Val Acc: {val_acc * 100:.4f}, Train Loss: {total_loss:.2f}, Val Loss: {val_loss:.4f}')

            logger.info(
                f'Epoch: {epoch:03d}, Train Acc: {acc * 100:.4f}, Val Acc: {val_acc * 100:.4f}, Train Loss: {total_loss:.2f}, Val Loss: {val_loss:.4f}')

        # Early stopper
        if enable_early_stopping:
            if early_stopper.early_stop(val_loss):
                break
    
    logger.info(f'L2 norm of the model parameters: {l2_norm(model=model)}')

    return model, \
        train_acc_list, \
        train_loss_list, \
        val_acc_list, \
        val_loss_list


@torch.no_grad()
def validate_model(model, loader, device, criterion, edge_feature_compact=True):
    """Evaluate the model on evaluation mode"""

    model.to(device)
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        data.to(device)
        with torch.cuda.amp.autocast():  # Mix-precision
            if edge_feature_compact:
                out = model(data.x.to(torch.float), data.edge_index,
                            data.edge_attr.to(torch.float), data.batch)
            else:
                out = model(data.x.to(torch.float), data.edge_index, data.batch)

            loss += criterion(out, data.y.flatten().float()) / len(loader)
            acc += accuracy(out, data.y.flatten().to(torch.int64)) / len(loader)

    model.train()  # setting model back to train mode
    return loss, acc


@torch.no_grad()
def test_model(model, loader, device, criterion, edge_feature_compact=True):
    """Evaluate the model on evaluation mode"""

    model.to(device)
    model.eval()
    loss = 0
    acc = 0
    roc_auc_score = 0
    avg_precision_score = 0
    auc_pr_curve = 0 # average over the entire dataset

    for data in loader:
        data.to(device)
        with torch.cuda.amp.autocast():  # Mix-precision
            if edge_feature_compact:
                out = model(data.x.to(torch.float), data.edge_index,
                            data.edge_attr.to(torch.float), data.batch)
            else:
                out = model(data.x.to(torch.float), data.edge_index, data.batch)

            loss += criterion(out, data.y.flatten().float()) / len(loader)
            acc += accuracy(out, data.y.flatten().to(torch.int64)) / len(loader)
            roc_auc_score += compute_roc_auc_score(out, data.y.flatten().to(torch.int64)) / len(loader)
            avg_precision_score += compute_avg_precision_score(out, data.y.flatten().to(torch.int64)) / len(loader)

            precision, recall, _ = compute_PR_curve(out, data.y.flatten().to(torch.int64))
            auc_pr_curve += auc(recall.cpu().numpy(), precision.cpu().numpy()) / len(loader)

    print(f'\nModel accuracy on test dataset: {acc * 100:.3f}\n')
    print(f'ROC-AUC Score for Test set: {roc_auc_score:.3f}\n')
    print(f'Average Precision Score for Test set: {avg_precision_score:.3f}\n')
    print(f'AUC-PR Score for Test set: {auc_pr_curve:.3f}\n')

    return acc, roc_auc_score, avg_precision_score, auc_pr_curve


def visualize_results(train_list, val_list, meta_data, path_to_saving_artifacts):
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
    # fig.show()
    fig.write_image(os.path.join(path_to_saving_artifacts,
                    f"images/{meta_data['image_data']}.png"))


def read_yaml_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = yaml.safe_load(f)

    return data


def write_yaml_file(path_to_save_file, data):
    with open(path_to_save_file, 'w') as f:
        yaml.dump(data, f)


def update_content(data_dict, eval_name, model_name):

    file_path = os.path.join(base_path_for_saving_artifacts, eval_name, 'eval.yaml')

    if os.path.exists(file_path):
        file_content = read_yaml_file(file_path)
    else:
        file_content = {}

    # Check the model exist or not
    if model_name in file_content.keys():
        file_content[model_name]['test_accuracy'].append(data_dict['test_accuracy'])
        file_content[model_name]['roc_auc_score'].append(data_dict['roc_auc_score'])
        file_content[model_name]['avg_precision_score'].append(data_dict['avg_precision_score'])
        file_content[model_name]['auc_pr_curve'].append(data_dict['auc_pr_curve'])
    else:
        file_content[model_name] = {}
        file_content[model_name]['test_accuracy'] = []
        file_content[model_name]['test_accuracy'].append(data_dict['test_accuracy'])
        file_content[model_name]['roc_auc_score'] = [] 
        file_content[model_name]['roc_auc_score'].append(data_dict['roc_auc_score'])
        file_content[model_name]['avg_precision_score'] = []
        file_content[model_name]['avg_precision_score'].append(data_dict['avg_precision_score'])
        file_content[model_name]['auc_pr_curve'] = []
        file_content[model_name]['auc_pr_curve'].append(data_dict['auc_pr_curve'])

    # write the content back to a file
    write_yaml_file(file_path, file_content)


##################################
############################################################################################


################### - Main Execution - #############################################################

def run_experiment(experiment_name, model, train_loader, val_loader, test_loader, epochs, metadata_for_experiment, edge_feature_compact=True, enable_early_stopping=True, eval_mode=False):
    """Run an experiment on given settings"""

    print('\n\n#################################### ******************************** #################################')
    print('Starting the experiment...')

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}\n')

    # Class weight
    print('Class_weight: ', metadata_for_experiment['dataset']['class_weight'], '\n')
    class_weight = torch.FloatTensor([metadata_for_experiment['dataset']['class_weight']])

    # Criterion for loss calculation
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight.to(device))

    path_to_saving_artifacts = _create_directories(experiment_name)
    seed_everything()  # apply seeding
    print('Model architecture:\n', model)

    logger = get_logger(os.path.join(path_to_saving_artifacts, 'log'))

    # Run the training process
    trained_model, \
        train_acc_list, \
        train_loss_list, \
        val_acc_list, \
        val_loss_list = train(model,
                              train_loader,
                              val_loader,
                              epochs,
                              edge_feature_compact,
                              path_to_saving_artifacts,
                              enable_early_stopping,
                              logger,
                              device,
                              criterion)

    # Testing accuracy for the trained model
    test_accuracy, roc_auc_score, avg_precision_score, auc_pr_curve = test_model(trained_model, test_loader, device, criterion, edge_feature_compact)

    if eval_mode:
        data = {}
        data['test_accuracy'] = test_accuracy
        data['roc_auc_score'] = roc_auc_score.detach().cpu().numpy().item()
        data['avg_precision_score'] = avg_precision_score.detach().cpu().numpy().item()
        data['auc_pr_curve'] = auc_pr_curve.item()

        model_name = experiment_name.split('/')[-1].split('_')[0]
        eval_name = experiment_name.split('/')[0]
        update_content(data_dict=data, eval_name=eval_name, model_name=model_name)
    

    # Save the accuracy for the trained model on the test datedge_feature_compactaset
    logger.debug(f'Accuracy on test dataset: {test_accuracy}')
    logger.debug(f'ROC-AUC Score on test dataset: {roc_auc_score:.3f}')
    logger.debug(f'Average-Precision Score on test dataset: {avg_precision_score:.3f}')
    logger.debug(f'AUC-PR Score on test dataset: {auc_pr_curve:.3f}')

    # Visualize the loss
    visualize_results(train_loss_list, val_loss_list,
                      {'title': 'Train and Validation Loss Distribution',
                       'xaxis_title': 'Epoch',
                       'yaxis_title': 'Loss',
                       'legend': 'Split',
                       'image_data': 'losses'},
                      path_to_saving_artifacts)
    # Visualize the accuracy
    visualize_results([acc * 100 for acc in train_acc_list],
                      [acc * 100 for acc in val_acc_list],
                      {'title': 'Train and Validation Accuracy Distribution',
                       'xaxis_title': 'Epoch',
                       'yaxis_title': 'Accuracy',
                       'legend': 'Split',
                       'image_data': 'train_val_accuracy'},
                      path_to_saving_artifacts)

    # Saving artifacts for the experiment: hyperparameters and dataset metadatas
    logger.debug(metadata_for_experiment)

    # Saving the trained model
    print('\nSaving the trained model...\n')
    # saving entire model
    torch.save(trained_model, os.path.join(path_to_saving_artifacts, 'models/entire_model.pt'))
    # saving the model for inference
    torch.save(trained_model.state_dict(), os.path.join(path_to_saving_artifacts, 'models/inference_model.pt'))

    print('#################################### ******************************** #################################')
    print('Experiment is completed!!!')

    return trained_model, test_accuracy
