import argparse
import json
import logging
import pathlib
import time
from pathlib import Path
from typing import List

import mlflow
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

from src.utils.util import save_args, safe_append_dataframe_to_file

logger = logging.getLogger(__name__)


def train(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: torch.cuda.device,
        args: argparse.Namespace
) -> None:
    """Train the model on data in the dataloader for the number of epochs stored in args"""
    # Store number of trainable parameters
    args.num_trainable_params = model.num_trainable_params

    logger.info(f'Number of trainable parameters in the model {args.num_trainable_params}')
    logger.info(f'\nTraining with args\n{args}\n')

    # Initialize predictions and one key metric
    predictions = dict()
    benchmark = {'train': args.benchmark_start_value, 'valid': args.benchmark_start_value}
    save_weights(model, -1, args)

    # train for a predefined number of epochs
    for epoch in range(args.epoch_start, args.epochs):
        args.epoch = epoch
        logger.info(f'\nEpoch\t {epoch}\n')

        # train for one epoch and evaluate on train
        predictions['train'], metrics = train_one_epoch(
                dataloader['train'], model, criterion, optimizer, scheduler, device, args)
        torch.cuda.empty_cache()
        # overwrite saved predictions for error analysis
        save_predictions(predictions['train'], metrics, benchmark, args.benchmark_name, args, 'train')

        # update benchmark score
        if metrics[f'{args.benchmark_name}_train'] > benchmark['train']:
            benchmark['train'] = metrics[f'{args.benchmark_name}_train']

        # evaluate on validation dataset, save validation predictions, update evaluation structure
        predictions['valid'], metrics = evaluate(dataloader['valid'], model, criterion, device, args, 'valid')
        save_predictions(predictions['valid'], metrics, benchmark, args.benchmark_name, args, 'valid')

        # store weights of the layers with weighted sums
        save_weights(model, epoch, args)

        if args.save_model:
            # Save the last trained model
            logging.info('Save the last trained model')
            last_model_dir = Path(args.output_dir) / 'checkpoint-last' / 'model.bin'
            mlflow.set_tag('model-checkpoint-last', str(last_model_dir))

            save_model_states_to_dir(last_model_dir, model, optimizer, scheduler, epoch)
            
        # update benchmark score (for early stopping and saving the best model)
        if metrics[f'{args.benchmark_name}_valid'] > benchmark['valid']:
            # save/update the best trained model and the best epoch number
            logging.info('Save/update the best trained model and the best epoch number')

            mlflow.log_metric('best_epoch', epoch, step=args.global_step)
            mlflow.log_metric('step_at_best_epoch', args.global_step, step=args.global_step)

            if args.save_model:
                best_model_dir = Path(args.output_dir) / f'checkpoint-best-{args.benchmark_name}' / 'model.bin'
                mlflow.set_tag('model-checkpoint-best', str(best_model_dir))
                save_model_states_to_dir(best_model_dir, model, optimizer, scheduler, epoch)
            benchmark['valid'] = metrics[f'{args.benchmark_name}_valid']

        # mlflow
        mlflow.log_metric('epoch', epoch, step=args.global_step)


def save_predictions(
        predictions: List[int],
        metrics: dict,
        benchmark_metrics: dict,
        benchmark_name: str,
        args: argparse.Namespace,
        mode: str
) -> None:
    """"Save predicted labels (`predictions`) to a txt file with a path specified in `args`"""
    save_labels_to_file(predictions, Path(args.output_dir) / 'predictions' / f'pred_{mode}_last.txt')
    if metrics[f'{benchmark_name}_{mode}'] > benchmark_metrics[mode]:
        save_labels_to_file(
            predictions, Path(args.output_dir) / 'predictions' / f'pred_{mode}_best_{benchmark_name}.txt')


def save_model_states_to_dir(
        model_path: pathlib.Path or str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epoch: int
) -> None:
    """Save the model parameters, epoch, and training state"""
    logging.info(f'Saving model and states to {model_path}')
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, model_path)


def save_labels_to_file(
        labels: list,
        path: pathlib.Path,
        task: str = 'multiclass_classification'
) -> None:
    """Save labels at `path`"""
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    labels = list(map(lambda x: str(x) + '\n', labels))

    if task == 'multiclass_classification':
        with open(path, 'w') as f:
            f.writelines(labels)
    elif task == 'multilabel_classification':
        raise NotImplementedError


def logits_to_labels(
        logits: np.array,
        task: str = 'multiclass_classification'
) -> List[int]:
    """Convert probabilities to labels by choosing the label with the highest probability"""
    if task == 'multiclass_classification':
        predictions = logits.argmax(-1).tolist()
        return predictions
    elif task == 'multilabel_classification':
        raise NotImplementedError


def train_one_epoch(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: torch.cuda.device,
        args: argparse.Namespace
) -> (List[int], dict):
    """Run one training epoch for the model and return predictions and other metrics
    on the train or validation set provided by the dataloader"""
    model.train()
    start_time = time.time()
    batch_iter = tqdm.tqdm(dataloader)
    num_batches = batch_iter.total
    average_loss_one_epoch = 0.
    true_labels = []
    logits = None

    for idx, batch in enumerate(batch_iter):
        args.global_step += 1
        optimizer.zero_grad()
        inputs, masks, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        # To device
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        # Golden labels remain on gpu
        true_labels += labels.squeeze(-1).tolist()

        logits_one_batch = model(inputs, masks)  # dimension: B x C; B - batch size, C - number of classes
        del inputs, masks
        logits = logits_one_batch if logits is None else torch.cat((logits, logits_one_batch), dim=0)

        loss = criterion(logits_one_batch.unsqueeze(-1), labels)
        del logits_one_batch, labels

        # batch_iter.set_postfix_str(f"loss: {loss.item(): .16f}")
        average_loss_one_epoch += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_max_norm)
        optimizer.step()
        scheduler.step()
        # mlflow
        mlflow.log_metric('loss_train', loss.item(), step=args.global_step)
        torch.cuda.empty_cache()

    # update learning rate schedule after each epoch
    scheduler.step()
    # stop counting time
    train_time = time.time() - start_time

    # convert logits to labels and evaluate metrics
    predictions = logits_to_labels(logits)
    del logits

    # calculate and save clf metrics and loss
    metrics = compute_metrics(predictions, true_labels, mode='train')
    average_loss_one_epoch = average_loss_one_epoch / num_batches
    metrics['avg_loss_train'] = average_loss_one_epoch
    metrics['time_train_min'] = train_time/60.

    # mlflow
    mlflow.log_metrics(metrics, step=args.global_step)

    return predictions, metrics


def save_weights(
        model: torch.nn.Module,
        epoch: int,
        args: argparse.Namespace
) -> None:
    """Save weights learned for the models that include a learnable weighted sum layer"""
    if 'w_' in args.combination_type:
        if not args.weights.exists():
            args.weights.mkdir(parents=True)
        for weight_type in ['w_layers', 'w_tokens']:
            weights = model.output_combination.w_layers if weight_type == 'w_layers' \
                else model.output_combination.w_tokens
            weights = weights.data.tolist()
            columns = ['epoch'] + list(map(lambda x: str(x), range(len(weights))))
            table = pd.DataFrame(data=0, columns=columns, index=[0])
            table.loc[0, 'epoch'] = epoch
            table.iloc[0, 1:] = weights
            safe_append_dataframe_to_file(table_path=args.weights / f'{weight_type}.csv', table=table)


def compute_metrics(
        y_pred: List[int],
        y_true: List[int],
        task: str = 'multiclass_classification',
        mode: str = 'train'
) -> dict:
    """Compute precision, recall, F1 and return a dictionary with these metrics"""
    metrics = dict()
    if task == 'multiclass_classification':
        metrics[f'acc_{mode}'] = accuracy_score(y_true, y_pred)
        metrics[f'balanced_acc_{mode}'] = balanced_accuracy_score(y_true, y_pred)
        for avg_type in ['micro', 'macro', 'weighted']:
            metrics[f'precision_{avg_type}_{mode}'] = precision_score(y_true, y_pred, average=avg_type)
            metrics[f'recall_{avg_type}_{mode}'] = recall_score(y_true, y_pred, average=avg_type)
            metrics[f'f1_{avg_type}_{mode}'] = f1_score(y_true, y_pred, average=avg_type)
        return metrics
    elif task == 'multilabel_classification':
        raise NotImplementedError


@torch.no_grad()
def evaluate(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.CrossEntropyLoss,
        device: torch.cuda.device,
        args: argparse.Namespace,
        mode: str
) -> (List[int], dict):
    """Evaluate the trained model on `mode` (validation, mainly: `valid`) data in a `dataloader`,
    and return predictions list and metrics dictionary, including loss and time for inference"""
    model.eval()
    start_time = time.time()
    batch_iter = tqdm.tqdm(dataloader)
    average_loss_one_epoch = 0
    num_batches = batch_iter.total
    logits = None
    golden_labels = []

    for idx, batch in enumerate(batch_iter):

        inputs, masks, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        logits_one_batch = model(inputs, masks)  # dimension: B x C; B - batch size, C - number of classes
        logits = logits_one_batch if logits is None else torch.cat((logits, logits_one_batch), dim=0)

        if mode == 'valid':
            loss = criterion(logits_one_batch.unsqueeze(-1), labels).item() if mode == 'valid' else 0
            batch_iter.set_postfix_str(f"{mode} loss: {loss:.16f}")
            average_loss_one_epoch += loss
            # mlflow
            mlflow.log_metric('loss_valid', loss)
        golden_labels += labels.squeeze(-1).tolist()
        torch.cuda.empty_cache()

    eval_time = time.time() - start_time
    predictions = logits_to_labels(logits)

    # calculate clf metrics
    metrics = compute_metrics(predictions, golden_labels, mode=mode)
    average_loss_one_epoch = average_loss_one_epoch / num_batches
    metrics[f'avg_loss_{mode}'] = average_loss_one_epoch
    metrics[f'time_{mode}_min'] = eval_time/60.

    # mlflow
    mlflow.log_metrics(metrics, step=args.global_step)

    return predictions, metrics


@torch.no_grad()
def evaluate_on_test(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        device: torch.cuda.device,
        args: argparse.Namespace,
        mode='test'
) -> None:
    """Evaluate the model on a (test) dataset in `dataloader` and store metrics in MLFlow"""
    benchmark = {mode: args.benchmark_start_value}

    predictions, metrics = evaluate(dataloader, model, None, device, args, mode)
    save_predictions(predictions, metrics, benchmark, args.benchmark_name, args, mode)
    # mlflow
    mlflow.log_metrics(metrics, step=args.global_step)
