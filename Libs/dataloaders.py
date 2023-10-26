################### - Imports - #############################################################
import torch
from torch_geometric.loader import DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Libs.datasets import pick_the_dataset
from Libs.utils import seed_dataloader
from Libs.splitting import scaffold_split

##################################
############################################################################################


################### - Helper Functions - #############################################################
def create_dataset_splits(dataset, metadata, split_type='random'):
    # Shuffle the dataset
    # dataset = dataset.shuffle()

    if split_type == 'random':
        train_dataset, test_dataset = train_test_split(
            dataset, test_size=0.2, random_state=42)
        train_dataset, val_dataset = train_test_split(
            train_dataset, test_size=0.25, random_state=42)

    elif split_type == 'scaffold': # Scaffold data splitting
        dataset_name = metadata['dataset_name']
        if dataset_name == 'BACE':
            smiles_list = pd.read_csv(f'./Data/{dataset_name.lower()}/raw/{dataset_name.lower()}.csv')['mol'].tolist()
        elif dataset_name == 'BBBP': #FIXME not working! 
            smiles_list = pd.read_csv(f'./Data/{dataset_name.lower()}/raw/{dataset_name}.csv')['smiles'].tolist()
        elif dataset_name == 'HIV': 
            smiles_list = pd.read_csv(f'./Data/{dataset_name.lower()}/raw/{dataset_name}.csv')['smiles'].tolist()        
        else:
            smiles_list = pd.read_csv(f'./Data/{dataset_name.lower()}/raw/{dataset_name.lower()}.csv')['smiles'].tolist()

        train_dataset, val_dataset, test_dataset = scaffold_split(dataset, smiles_list, task_idx=None, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(val_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset

def get_class_weights(dataset):
    """Get the class weight for the class imbalance
    
    Args:
        dataset: original dataset: torch_geometric.dataset

    Return:
        weight: final weight to be applied for the loss function (float)
    """    

    for i in range(len(dataset)):
        y_i = dataset[i].y.cpu().detach().numpy()
        y = y_i if i ==0 else np.vstack([y,dataset[i].y.cpu().detach().numpy()]) 

    # Creating the dataframe
    y_flat = y.flatten()
    df_y = pd.DataFrame({'Target': y_flat})

    # Get the value count / class occurance
    count_arr = list(df_y.value_counts())  # Like [1567, 483]
    # # Take the weight by dividing the highest occuring class by minimum class
    weight = max(count_arr) / min(count_arr)

    return weight

##################################
############################################################################################


################### - Main Execution - #############################################################
def create_dataloaders(args):
    """Create required dataloaders for the given dataset
        Args:
            args: for picking the dataset: args.dataset_name, args.preprocess, args.do_analysis
        Returns:
            train_loader, val_loader, test_loader
    """

    dataset, metadata = pick_the_dataset(args)
    train_dataset, val_dataset, test_dataset = create_dataset_splits(dataset, metadata, args['split_type'])

    # Class weight
    class_weight = get_class_weights(dataset=dataset)
    metadata['class_weight'] = class_weight

    seed_worker, generator = seed_dataloader()

    train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=False, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(
        val_dataset, batch_size=args['batch_size'], shuffle=False, worker_init_fn=seed_worker, generator=generator)
    test_loader = DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False, worker_init_fn=seed_worker, generator=generator)

    return train_loader, val_loader, test_loader, metadata
