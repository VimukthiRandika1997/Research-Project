################### - Imports - #############################################################
import os
import time

from torch_geometric.datasets import TUDataset, MoleculeNet
from Libs.dataset_featurizer import MyMoleculeDataset


import plotly.express as px

##################################
############################################################################################


################### - Paths - #############################################################
path_to_save_dataset = './Data'

##################################
############################################################################################


################### - Helper Functions - #############################################################
def _create_directories():
    os.makedirs(path_to_save_dataset, exist_ok=True)


def prepare_dataset(dataset):
    """Do some preprocessing on the selected dataset"""
    return dataset


def analysize_dataset():
    """Do some analysis on the selected dataset"""
    pass

##################################
############################################################################################


################### - Main Execution - #############################################################

def pick_the_dataset(args):
    """Pick the desired dataset for the evaluation process
        Args:
            args.dataset_name: str: datset name
            args.preprocess: bool: is requred the preprocess step or not 
            args.do_analysis: bool: is analysis step required or not

    """

    _create_directories()

    # Metadata related the dataset
    metadata = {}

    if args['dataset_name'] == 'MUTAG':
        dataset = TUDataset(root='./Data', name='MUTAG')

    elif args['dataset_name'] == 'BBBP':
        dataset = MyMoleculeDataset('./Data', filename='BBBP.csv')

    elif args['dataset_name'] == 'HIV':
        dataset = MyMoleculeDataset('./Data', filename='HIV.csv')
        
    elif args['dataset_name'] == 'BACE':
        dataset = MyMoleculeDataset('./Data', filename='bace.csv')

    else:
        raise ValueError(
            'Unknown dataset name, please specify a valid dataset')

    print(f'Dataset: {dataset}:')

    print()
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of edge features: {dataset.num_edge_features}')

    # Storing metadata
    metadata['dataset_name'] = args['dataset_name']
    metadata['num_node_features'] = dataset.num_features
    metadata['num_edge_features'] = dataset.num_edge_features
    metadata['num_cls'] = dataset.num_classes
    metadata['num_graphs'] = len(dataset)

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================\n')

    # Gather some statistics about the first graph.
    print('Some statistics about the first graph...')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print('=============================================================\n')

    # if args['preprocess']:
    #     preprocessed_dataset = prepare_dataset()
    #     return preprocessed_dataset

    # if args['do_analysis']:
    #     analysize_dataset()

    return dataset, metadata
