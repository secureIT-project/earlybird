import logging
import time
import warnings
from pathlib import Path

import mlflow
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (get_linear_schedule_with_warmup)

from src.dataset.dataset import CodeClassificationDataset
from src.model.model import EncoderWithLayerCombination
from src.utils.batch_processing import train, evaluate_on_test
from src.utils.util import get_time_hh_mm_ss, set_seed, seed_worker, setup_args_and_paths

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Track time
    time_start = time.time()

    args, config = setup_args_and_paths(time_start)

    # You can optionally organize runs into experiments, which group together runs for a specific task.
    mlflow.set_experiment(f'{args.dataset_name}_{args.model_name}')
    mlflow.log_params(vars(args))

    # Logger
    log_filepath = Path(config['project_path']) / 'logs' / \
                   f'{args.slurm_job_id}_{args.dataset_name}_{args.model_name}_{args.combination_type}_' \
                   f'layer_{args.hidden_layer_to_use}_exp{args.experiment_no}.txt'
    log_filepath.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO, filename=log_filepath)
    logger = logging.getLogger(__name__)

    args.global_step = 0

    # Initialize cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f'Device: {device}')

    # Fix random state
    g = set_seed(args.seed)

    # Small subsets of data for debug
    if args.debug:
        args.batch_size = 4

    dataset = dict()
    dataloader = dict()
    model = None

    if args.train:
        dataset['train'] = CodeClassificationDataset(args, config, mode='train')
        dataset['valid'] = CodeClassificationDataset(args, config, mode='valid')

        dataloader['train'] = DataLoader(
            dataset=dataset['train'], batch_size=args.batch_size, worker_init_fn=seed_worker,
            generator=g, shuffle=True, drop_last=True)
        dataloader['valid'] = DataLoader(
            dataset=dataset['valid'], batch_size=args.batch_size, shuffle=False, drop_last=False)

        args.num_classes = len(set(dataset['train'].labels))
        model = EncoderWithLayerCombination(args)
        model.to(device)

        if torch.__version__.startswith('2.'):
            logger.info(f'Pytorch version: {torch.__version__}\nTorch 2.0 acceleration with torch.compile in train')
            print(f'Pytorch version: {torch.__version__}\nTorch 2.0 acceleration with torch.compile in train')
            model = torch.compile(model)

        mlflow.log_param('num_trainable_parameters', model.num_trainable_params)

        # optimizer and scheduler setup is taken from https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Defect-detection/code/run.py#L135
        # Prepare optimizer, schedule (linear warmup and decay) and loss function calculation
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(args.learning_rate), eps=float(args.adam_epsilon))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps),
                                                    num_training_steps=int(args.num_train_optimization_steps))

        if args.use_class_weights_in_loss:
            class_weights = dataset['train'].class_weights()
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, weight=torch.Tensor(class_weights))
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Train and evaluate the model on train and validation dataset
        logger.info(f"Preprocessing time: {int(time.time() - time_start)}")
        logger.info('Training and evaluating each epoch')

        train(dataloader, model, criterion, optimizer, scheduler, device, args)

    if args.test:
        dataset['test'] = CodeClassificationDataset(args, config, mode='test')
        dataloader['test'] = DataLoader(
            dataset=dataset['test'], batch_size=args.batch_size, shuffle=False, drop_last=False)

        if not args.train:
            num_classes = len(dataset['test'].label_encoder.classes_)
            args.num_classes = num_classes

            # configure paths and read arguments that the model was trained with
            checkpoint_best_path = Path(args.output_dir) / f'checkpoint-best-{args.benchmark_name}' \
                if args.eval_model_path is None else Path(
                args.eval_model_path) / f'checkpoint-best-{args.benchmark_name}'

            model = EncoderWithLayerCombination(args)
            checkpoint = torch.load(checkpoint_best_path / 'model.bin')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            if torch.__version__.startswith('2.'):
                logger.info(f'Pytorch version: {torch.__version__}\nTorch 2.0 acceleration with torch.compile in test')
                print(f'Pytorch version: {torch.__version__}\nTorch 2.0 acceleration with torch.compile test')
                model = torch.compile(model)

        evaluate_on_test(dataloader['test'], model, device, args, 'test')

    logger.info('Total time spent (below)')
    time_stop = time.time()
    total_time = get_time_hh_mm_ss(int(time_stop - time_start))
    mlflow.log_param('total_time_hhmmss', total_time)
    mlflow.end_run()
    logger.info(f"End of execution.")
    logging.shutdown()


