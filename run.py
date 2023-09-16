################### - Imports - #############################################################
from Libs.dataloaders import create_dataloaders
from Libs.models import get_all_models
from Libs.train import run_experiment

##################################
############################################################################################


################### - Arguments for running experiments - ############################################################
# Datasets for experiments
datasets = ['MUTAG', '']

# Model buleprints
model_blueprints = get_all_models()

# Baseline hyperparameters for each of the model blueprints
model_hyperparameter_list = {

    'hyperparameters_for_EIN_model': {
        'dim_h': 64,
        'num_heads': 16,
        'eps': 1

    },

    'hyperparameters_for_GCN_model': {
        'dim_h': 64
    },

    'hyperparameters_for_GAT_model': {
        'dim_h': 64,
        'num_heads': 16,
    },

    'hyperparameters_for_GIN_model': {
        'dim_h': 64,
        'train_eps': True,
        'eps': 1
    }

}

# Arguments for dataset creation
args = {
    'dataset_name': 'MUTAG',
    'batch_size': 64
}


##################################
############################################################################################

################### - Main Execution - ############################################################


def run():
    train_loader, \
        val_loader, \
        test_loader, \
        metadata = create_dataloaders(args)

    # Storing created models for training
    model_list = []

    # Hyperparameter updating based on the dataset metadata
    assert len(model_hyperparameter_list) == len(
        model_blueprints), 'Length should be equal!!!'

    # Does not contain edge features, hence those are excluded
    excluded_models = set(['GCN', 'GIN'])

    for key, model_blueprint in zip(model_hyperparameter_list.keys(), model_blueprints):
        model_hyperparameter_list[key]['input_dim'] = metadata['num_node_features']
        model_hyperparameter_list[key]['final_dim'] = metadata['num_cls']

        # avoid models without edge features
        if not (key.split('_')[-2] in excluded_models):
            print(key)
            model_hyperparameter_list[key]['edge_dim'] = metadata['num_edge_features']

        # Creating the model
        model = model_blueprint(**model_hyperparameter_list[key])
        model_list.append(model)

    # Running the experiments
    for i, (model, key) in enumerate(zip(model_list, model_hyperparameter_list.keys())):

        metadata_for_experiment = {
            'hyperparameters': model_hyperparameter_list[key],
            'dataset': metadata
        }
        # avoid models without edge features
        model_name = key.split('_')[-2]
        if not (model_name in excluded_models):

            run_experiment(experiment_name=f'{model_name}_exp' + str(i),
                           model=model,
                           train_loader=train_loader,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           epochs=300,
                           metadata_for_experiment=metadata_for_experiment,
                           #    enable_early_stopping=False
                           )
        else:

            run_experiment(experiment_name=f'{model_name}_exp' + str(i),
                           model=model,
                           train_loader=train_loader,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           epochs=300,
                           metadata_for_experiment=metadata_for_experiment,
                           edge_feature_compact=False
                           #    enable_early_stopping=False
                           )


if __name__ == '__main__':
    run()
