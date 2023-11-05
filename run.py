################### - Imports - #############################################################
from Libs.dataloaders import create_dataloaders
from Libs.models import get_all_models
from Libs.train import run_experiment

import datetime

##################################
############################################################################################


################### - Arguments for running experiments - ############################################################
EPOCHS=500
BATCH_SIZE=64
DIM_H=64
NUM_HEADS=16
TRAIN_EPS=True
EPS=1

# Datasets for experiments
datasets = ['MUTAG', 'BBBP', 'Tox21', 'HIV', 'PROTEINS', 'BACE']

# Arguments for dataset creation
args = {
    # 'dataset_name': 'MUTAG',
    # 'dataset_name': 'BACE',
    'dataset_name': 'BBBP',
    # 'dataset_name': 'HIV',
    'batch_size': BATCH_SIZE,
    'split_type': 'random',
    # 'split_type': 'scaffold',
    'epochs': EPOCHS,
    'eval_mode': False
}


# Model buleprints
model_blueprints = get_all_models()

# Baseline hyperparameters for each of the model blueprints
model_hyperparameter_list = {

    'hyperparameters_for_EIN_model': {
        'dim_h': DIM_H,
        'num_heads': NUM_HEADS,
        'train_eps': TRAIN_EPS,
        'eps': EPS

    },

    'hyperparameters_for_EINv2_model': {
        'dim_h': DIM_H,
        'num_heads': NUM_HEADS,
        'train_eps': TRAIN_EPS,
        'eps': EPS

    },     

    'hyperparameters_for_EINv3_model': {
        'dim_h': DIM_H,
        'num_heads': NUM_HEADS,
        'train_eps': TRAIN_EPS,
        'eps': EPS

    },     

    #  'hyperparameters_for_EINv4_model': {
    #     'dim_h': DIM_H,
    #     'num_heads': NUM_HEADS,
    #     'train_eps': TRAIN_EPS,
    #     'eps': EPS

    # }, 

    'hyperparameters_for_EINv5_model': {
        'dim_h': DIM_H,
        'num_heads': NUM_HEADS,
        'train_eps': TRAIN_EPS,
        'eps': EPS,
        'share_weights': True

    },

    'hyperparameters_for_GCN_model': {
        'dim_h': DIM_H
    },

    'hyperparameters_for_GAT_model': {
        'dim_h': DIM_H,
        'num_heads': NUM_HEADS,
    },

    'hyperparameters_for_GIN_model': {
        'dim_h': DIM_H,
        'train_eps': TRAIN_EPS,
        'eps': EPS
    },

    'hyperparameters_for_GINE_model': {
        'dim_h': DIM_H,
        'train_eps': TRAIN_EPS,
        'eps': EPS
    },  

}


##################################
############################################################################################

################### - Main Execution - ############################################################


def run(args):
    train_loader, \
        val_loader, \
        test_loader, \
        metadata = create_dataloaders(args)

    # Storing created models for training
    model_list = []

    # Hyperparameter updating based on the dataset metadata
    assert len(model_hyperparameter_list) == len(model_blueprints), 'Length should be equal!!!'

    # Does not contain edge features, hence those are excluded
    excluded_models = set(['GCN', 'GIN',])

    for key, model_blueprint in zip(model_hyperparameter_list.keys(), model_blueprints):
        model_hyperparameter_list[key]['input_dim'] = metadata['num_node_features']

        if metadata['num_cls'] == 2:
            model_hyperparameter_list[key]['final_dim'] = 1
        else:
            model_hyperparameter_list[key]['final_dim'] = metadata['num_cls']

        # avoid models without edge features
        if not (key.split('_')[-2] in excluded_models):
            # print(key)
            model_hyperparameter_list[key]['edge_dim'] = metadata['num_edge_features']

        # Creating the model
        model = model_blueprint(**model_hyperparameter_list[key])
        model_list.append(model)

    current_time = datetime.datetime.now()

    # Running the experiments
    for i, (model, key) in enumerate(zip(model_list, model_hyperparameter_list.keys())):

        metadata_for_experiment = {
            'hyperparameters': model_hyperparameter_list[key],
            'dataset': metadata
        }
        # avoid models without edge features
        model_name = key.split('_')[-2]
        if not (model_name in excluded_models):

            if args['eval_mode']:
                run_experiment(experiment_name=f"{args['eval_name']}/{args['dataset_name']}/exp_{str(current_time)}/{model_name}_exp",
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            epochs=args['epochs'],
                            metadata_for_experiment=metadata_for_experiment,
                            enable_early_stopping=False,
                            eval_mode=args['eval_mode']
                            )
            else:
                run_experiment(experiment_name=f"{args['dataset_name']}/exp_{str(current_time)}/{model_name}_exp",
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            epochs=args['epochs'],
                            metadata_for_experiment=metadata_for_experiment,
                            enable_early_stopping=False,
                            eval_mode=args['eval_mode']
                )
        else:

            if args['eval_mode']:
                run_experiment(experiment_name=f"{args['eval_name']}/{args['dataset_name']}/exp_{str(current_time)}/{model_name}_exp",
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            epochs=args['epochs'],
                            metadata_for_experiment=metadata_for_experiment,
                            edge_feature_compact=False,
                            enable_early_stopping=False,
                            eval_mode=args['eval_mode']
                            )
            else:
                run_experiment(experiment_name=f"{args['dataset_name']}/exp_{str(current_time)}/{model_name}_exp",
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            epochs=args['epochs'],
                            metadata_for_experiment=metadata_for_experiment,
                            edge_feature_compact=False,
                            enable_early_stopping=False,
                            eval_mode=args['eval_mode']
                )


if __name__ == '__main__':
    run(args)
