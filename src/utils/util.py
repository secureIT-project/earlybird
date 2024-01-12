import argparse
import datetime
import json
import jsonlines
import logging
import os
import pathlib
import random
import sys
import torch
import yaml

import numpy as np
import pandas as pd

from collections import OrderedDict
from datetime import timedelta
from typing import Dict, List


def setup_args_and_paths(time_start: float):
    """Update and save arguments for the hyperparameters used"""
    # Parse command line and yaml arguments
    args = parse_args(sys.argv[1:])
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Manual update of experiment parameters
    args.experiment_id = custom_experiment_id(args, time_start)

    args.slurm_job_id = os.environ.get('SLURM_JOB_ID')

    # Output folder for a specific run
    args.output_dir = pathlib.Path(config['project_path']) / config['output'] / 'runs' / f'{args.experiment_id}'

    # Dump the copy of the full config and full args before they are all changed to something else
    (args.output_dir / 'params').mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / 'params' / 'config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    save_args(args.output_dir / 'params' / 'args.json', vars(args))

    # General output folders
    set_output_paths_in_args(args, config)
    return args, config


def custom_experiment_id(args: argparse.Namespace, time_start: float) -> str:
    """Create an informative experiment ID with hyperparameter variables"""
    experiment_id = f'epoch_{args.epochs}_{args.dataset_name}_{args.model_name}_'
    experiment_id += f'{args.combination_type}_layer_no_{args.hidden_layer_to_use}_' \
        if 'one_layer' in args.combination_type else f'{args.combination_type}_'
    experiment_id += f'clf_{args.clf_architecture}_layer_norm_{args.add_layer_pre_normalization}_' \
                     f'freeze_embeddings_{args.freeze_embeddings}_freeze_base_model_{args.freeze_base_model}_' \
                     f'lr_{args.learning_rate}_B_{args.batch_size}___' \
                     f'{datetime.datetime.fromtimestamp(time_start).strftime("%d-%m-%Y-%H_%M_%S_%f")[:-3]}'
    return experiment_id


def set_output_paths_in_args(args: argparse.Namespace, config: dict) -> None:
    """Update args to include paths to files with logs, predictions and model checkpoints"""
    args.tables = pathlib.Path(config['project_path']) / config['tables']
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures = pathlib.Path(config['project_path']) / config['figures']
    args.figures.mkdir(parents=True, exist_ok=True)
    args.logs = pathlib.Path(config['project_path']) / config['logs']
    args.logs.mkdir(parents=True, exist_ok=True)
    args.weights = pathlib.Path(args.output_dir) / 'weights'
    args.weights.mkdir(parents=True, exist_ok=True)
    try:
        args.model_path = pathlib.Path(config['project_path']) / config['model_path']
    except KeyError:
        args.model_path = pathlib.Path(config['project_path']) / args.model_path

    try:
        args.tokenizer_path = pathlib.Path(config['project_path']) / config['tokenizer_path']
    except KeyError:
        args.tokenizer_path = pathlib.Path(config['project_path']) / args.tokenizer_path


def get_time_hh_mm_ss(sec: float) -> str:
    """Convert timestamp to human-readable format (hh:mm:ss)"""
    logging.info(f'Time in seconds: {sec}')

    # create timedelta and convert it into string
    td_str = str(timedelta(seconds=sec))

    # split string into individual component
    x = td_str.split(':')
    logging.info(f'Time in hh:mm:ss: {x[0]}:{x[1]}:{x[2]}')
    return f'{x[0]}:{x[1]}:{x[2]}'


def parse_args(args_in: List[str]) -> argparse.Namespace:
    """Parse command line arguments, see help for more information"""
    parser = argparse.ArgumentParser()

    # ---------------------------
    # Run
    # ---------------------------
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--experiment_no', default=-1, type=int)
    parser.add_argument('--config_path', type=str, help='Path to config.yaml')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--plot_run', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    
    # Optimization and learning parameters
    parser.add_argument('-lr', '--learning_rate', default=1e-5, type=float)
    parser.add_argument('--use_class_weights_in_loss', default=False, action='store_true')
    parser.add_argument('-warmup', '--warmup_steps', default=10000, type=int)
    parser.add_argument('-batch', '--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    # Evaluation strategy
    parser.add_argument('--benchmark_name', type=str, default='f1_weighted',
                        help='Main metric to compare results and store the best performing model')
    parser.add_argument('--benchmark_start_value', default=-1., type=float)
    # ---------------------------

    # ---------------------------
    # Model
    # ---------------------------
    parser.add_argument('-clf', '--clf_architecture', default='one_linear_layer', type=str,
                        choices=['one_linear_layer', 'roberta_classification_head'])

    parser.add_argument('--classifier_dropout', default=0.1, type=float)

    # Combination of layers
    parser.add_argument('--combination_type', default='w_sum_tokens_w_sum_layers', type=str,
                        choices=[
                            'cutoff_layers_one_layer_cls',
                            'one_layer_w_sum_tokens',
                            'one_layer_max_pool_tokens',
                            'one_layer_cls',
                            'w_sum_tokens_w_sum_layers',
                            'max_pool_layers_w_sum_tokens',
                            'w_sum_layers_w_sum_tokens',
                            'max_pool_tokens_w_sum_layers',
                            'max_pool_layers_max_pool_tokens',
                            'w_sum_cls',
                            'max_pool_cls',
                            'last_layer_cls'
                        ])
    parser.add_argument('--hidden_layer_to_use', default=12, type=int, choices=list(range(1, 13, 1)))

    # Optional configurations
    parser.add_argument('-layer_norm', '--add_layer_pre_normalization', default=False, action='store_true')
    # ---------------------------

    # ---------------------
    # MODEL - FIXED PARAMETERS
    # ---------------------
    parser.add_argument('--embedding_dim', default=768, type=int)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--max_length', default=512, type=int)

    parser.add_argument('--not_special_tokens_to_zero', default=False, action='store_true')
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)

    parser.add_argument('--model_name', type=str, default='codebert')
    parser.add_argument('--model_path', type=str, default='checkpoints/reused/model/codebert-base',
                        help='Relative path from the top project folder, one above src, to the saved model.')
    parser.add_argument('--tokenizer_path', type=str, default='checkpoints/reused/model/codebert-base',
                        help='Relative path from the top project folder, one above src, to the saved tokenizer, '
                             'usually in the same set of files as the model.')

    # ---------------------
    # OPTIMIZER
    # ---------------------
    # defaults from CodeBERT for code search
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    # warmup_steps       : 10000
    parser.add_argument('--num_train_optimization_steps', default=100000, type=int)
    parser.add_argument('--clip_grad_max_norm', default=0.1, type=float)

    # ---------------------
    # RUN - FIXED PARAMETERS
    # ---------------------
    parser.add_argument('--train_steps', default=None)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--early_stop_count', default=-1, type=int,
                        help='count of epochs or train_steps for non-improving metric to stop the training, '
                             '-1 for disabling')
    parser.add_argument('--train', default=False, action='store_true',
                        help='train model and evaluate on validation set')
    parser.add_argument('--test', default=False, action='store_true', help='evaluate the model on the test set')
    parser.add_argument('--do_not_build_label_encoder', default=False, action='store_true',
                        help='label encoder is updated on each run if this argument is not set')
    parser.add_argument('--freeze_embeddings', default=False, action='store_true')
    parser.add_argument('--freeze_base_model', default=False, action='store_true')

    # ---------------------------
    # Data
    # ---------------------------
    parser.add_argument('--dataset_stats', default=False, action='store_true')
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--dataset_name', default='devign', type=str,
                        choices=['devign', 'reveal', 'break_it_fix_it', 'exception'])
    parser.add_argument('--ratio_train', default=0.8, type=float)
    parser.add_argument('--ratio_valid', default=0.1, type=float)
    parser.add_argument('--ratio_test', default=0.1, type=float)
    parser.add_argument('--underrepresented_threshold', default=0.0, type=float)
    parser.add_argument('--preprocess_python_data', default=False, action='store_true')
    parser.add_argument('--custom_split', default=False, action='store_true')
    parser.add_argument('--shrink_code', default=False, action='store_true')
    parser.add_argument('--keep_formatting', default=False, action='store_true')

    return parser.parse_args(args_in)


def set_seed(seed_value: int) -> torch.Generator:
    """Fix random seed for torch, numpy, and torch generator"""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed_value)
    return g


def seed_worker() -> None:
    """Fix the worker seed (for shuffling data in dataloader later)"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_args(filepath: pathlib.Path or str, source_dict: Dict) -> None:
    """Save hyperparameters of the current experiment to a file"""
    res_dict = dict()
    for key, value in source_dict.items():
        res_dict[key] = value \
            if not type(value) in [pathlib.PosixPath, pathlib.Path, pathlib.PurePath, pathlib.PurePosixPath] \
            else str(value)
    with open(filepath, 'w') as f:
        json.dump(OrderedDict(sorted(res_dict.items())), f, indent=2)


def safe_append_dataframe_to_file(table_path: pathlib.Path, table: pd.DataFrame) -> None:
    """
    Append one line to the dataframe stored in a file without overwriting it
    or store to a file if it does not exist
    """
    if table_path.exists():
        global_eval_df = pd.read_csv(filepath_or_buffer=table_path, index_col=0)
        global_eval_df = global_eval_df.append(table, ignore_index=True)
        global_eval_df.drop_duplicates(ignore_index=True, inplace=True)
        global_eval_df.to_csv(path_or_buf=table_path)
    else:
        table.to_csv(path_or_buf=table_path)


def safe_append_dict_entries_to_json(json_path: pathlib.Path or str, data: dict) -> None:
    """Append data from a file to the existing dataset. Used for datasets split into different jsonlines files."""
    if json_path.exists():
        with open(json_path, 'r') as f:
            stored_data = json.load(f)
        stored_data.update(data)
    else:
        if not json_path.parent.exists():
            json_path.parent.mkdir(parents=True)
        stored_data = data
    with open(json_path, 'w') as f:
        json.dump(stored_data, f, indent=2)


def load_jsonl(input_path: pathlib.Path or str) -> list:
    """Read list of objects from a json-lines file."""
    data = []
    with jsonlines.open(input_path, mode='r') as f:
        for line in f.iter():
            data.append(line)
    logging.info(f'Read {len(data)} records from {input_path}')
    return data
