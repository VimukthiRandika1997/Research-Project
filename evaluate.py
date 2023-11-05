import os
import sys
import datetime
import yaml
import argparse

from run import run as one_run

################### - Arguments for running experiments - ############################################################
EPOCHS=500
BATCH_SIZE=64
DIM_H=64
NUM_HEADS=16
TRAIN_EPS=True
EPS=1

# Evaluation name
eval_name = f'eval_{datetime.datetime.now()}'

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
    'eval_mode': True,
    'eval_name': eval_name
}
######################################################################################################################
base_path_for_saving_artifacts = './Runs'

def read_yaml_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = yaml.safe_load(f)

    return data


def write_yaml_file(path_to_save_file, data):
    with open(path_to_save_file, 'w') as f:
        yaml.dump(data, f)


def compute_average(data_arr):
    avg = 0
    for value in data_arr:
        avg += value

    return avg / len(data_arr)


def evaluate_gnn_models(max_iter=10):
    """Evaluate GNN models for `max_iter` times and take the average results"""

    result = {}

    # Run for `max_iter` times
    for i in range(max_iter):
        one_run(args)

    # Read the eval_file
    path_to_eval_file = os.path.join(base_path_for_saving_artifacts, eval_name, 'eval.yaml')
    data = read_yaml_file(path_to_eval_file)

    # Take the average results    
    for model_name in data.keys():
        model_res_dict = {}

        metric_arr = data[model_name].keys()
        for metric in metric_arr:
            metric_value_arr = data[model_name][metric]
            avg = compute_average(metric_value_arr)
            model_res_dict[metric] = avg

        result[model_name] = model_res_dict

    # Save the final result in a yaml file
    path_save_final_res = os.path.join(base_path_for_saving_artifacts, eval_name, 'final_result.yaml')
    write_yaml_file(path_save_final_res, result)

if __name__ == '__main__':
    evaluate_gnn_models()