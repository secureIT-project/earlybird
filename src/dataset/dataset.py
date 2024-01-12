import argparse
import collections
import itertools
import logging
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn import preprocessing
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.utils.util import safe_append_dataframe_to_file, load_jsonl
from src.visualization.visualize import custom_barplot

logger = logging.getLogger(__name__)


def build_or_load_cached_label_encoder(
        args: argparse.Namespace,
        config: dict,
        labels: List[str]
) -> preprocessing.LabelEncoder:
    """Create a new joblib label encoder or load an existing one
    using the paths from `config`"""
    le_path = Path(config['project_path']) / config[args.dataset_name]['label_encoder_path']
    if le_path.exists() and args.do_not_build_label_encoder:# or mode != 'train': # do not re-create le
        logging.info(f'Loading label encoder from {le_path}')
        le = joblib.load(le_path)
    else:
        logging.info(f'Creating a new label encoder at: {le_path}')
        le = preprocessing.LabelEncoder()
        le = le.fit(np.array(labels))
        logging.info(f'Class labels: \n{le.classes_}')
        le_path.parents[0].mkdir(parents=True, exist_ok=True)
        joblib.dump(le, le_path)
    return le


def read_classification_data_from_txt(
        args: argparse.Namespace,
        config: dict,
        split: str
) -> (list, list):
    """Load data stored in txt"""
    logging.info(f'Read {split} data from txt')
    with open(Path(config['project_path']) / vars(args)[f'{split}_data_path'], 'r') as f:
        corpus = list(map(lambda x: x.strip(), f.readlines()))
    with open(Path(config['project_path']) / vars(args)[f'{split}_labels_path'], 'r') as f:
        labels = list(map(lambda x: x.strip(), f.readlines()))
    return corpus, labels


def read_classification_data_from_jsonl(
        filepath: str or Path,
        split: str
) -> (list, list):
    """Load data stored in jsonlines"""
    logging.info(f'Read {split} data from {filepath}')
    corpus = []
    labels = []
    data = load_jsonl(filepath)
    for line in data:
        corpus.append(line['src'])
        labels.append(line['label'])
    return corpus, labels


def read_input_data(
        args: argparse.Namespace,
        config: dict,
        split: str
) -> (List[str], List[int]):
    """Read a dataset of code snippets and labels"""
    data_reader = {
        'txt': read_classification_data_from_txt,
        'jsonl': read_classification_data_from_jsonl
    }
    input_filepath = \
        {'train': Path(config['project_path']) / config[args.dataset_name]['processed']['train_data_path'],
         'valid': Path(config['project_path']) / config[args.dataset_name]['processed']['valid_data_path'],
         'test': Path(config['project_path']) / config[args.dataset_name]['processed']['test_data_path']}[split]
    corpus, labels = data_reader[input_filepath.suffix[1:]](input_filepath, split)

    # Take small subsets: train/valid/test = 80 / 10 / 10 or even 16 / 2 / 2
    if args.debug:
        logging.info(f'Debug mode on, using a small subset of data')
        if args.device == "cuda":
            subset = int(80 * int(split == 'train') + 10)
        else:
            subset = int(14 * int(split == 'train') + 2)
        corpus = corpus[:subset]
        labels = labels[:subset]
    return corpus, labels


class CodeClassificationDataset(Dataset):
    def __init__(self, args, config, mode='train'):
        self.corpus, self.labels = read_input_data(args, config, mode)
        logging.info(f'Loading tokenizer from {args.tokenizer_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

        if not self.tokenizer.pad_token:
            # default special token in GPT2TokenizerFast is <|endoftext|> - required for Starencoder
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        self.label_encoder = build_or_load_cached_label_encoder(args, config, self.labels)
        self.max_length = args.max_length
        if args.dataset_stats:
            self.compute_stats(args, mode)

    def __getitem__(self, index):
        sample = self.corpus[index]
        encoded_labels = self.label_encoder.transform([self.labels[index]])
        encoding = self.tokenizer(
            sample,
            is_split_into_words=False,
            return_tensors='pt',
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length)
        # Turn everything into PyTorch tensors
        item = {key: torch.LongTensor(val) for key, val in encoding.items()}
        item['labels'] = torch.LongTensor(encoded_labels)
        return item

    def __len__(self):
        return len(self.corpus)

    def class_weights(self) -> List[float]:
        """Calculate weights for each class based on class size to count for unbalanced classes in loss, for example"""
        class_distribution = collections.Counter(list(itertools.chain(self.labels)))
        corpus_len = self.__len__()
        class_weights = [class_distribution[self.label_encoder.inverse_transform([i])[0]]/corpus_len \
            for i in range(len(self.label_encoder.classes_))]
        return class_weights

    def compute_stats(self, args: argparse.Namespace, split: str) -> None:
        """Describe dataset characteristics (class sizes and token counts) and save the in files specified in `args`"""
        count = collections.Counter(list(itertools.chain(self.labels)))
        logger.info(f'{split}: {count}')
        class_table = pd.DataFrame.from_dict(dict(count), orient='index', columns=['num_samples'])
        class_table.to_csv(args.tables / f'{args.dataset_name}_{args.pl}_{split}_class_disctribution.csv')
        custom_barplot(
            labels=class_table.index.values,
            values=class_table.num_samples.values,
            xlabel='Class labels',
            ylabel='',
            ylim=(0., float(class_table.num_samples.max()) * 1.05),
            title=f'Class label distribution in {args.dataset_name} {args.pl} {split}',
            filepath=(args.figures / f'{args.dataset_name}_{args.pl}_{split}_class_disctribution.png')
        )
        stats_table = pd.DataFrame(
            data=0,
            columns=['mode', 'dataset_name', 'pl',
                     'num_samples', 'num_tokens_avg', 'num_tokens_max', 'num_tokens_min', 'num_tokens_std'],
            index=[0]
        )
        stats_table.loc[0, ['dataset_name', 'pl', 'mode']] = args.dataset_name, args.pl, split
        stats_table.loc[0, 'num_samples'] = len(self.corpus)
        num_tokens = np.array([len(self.tokenizer.tokenize(c)) for c in self.corpus])
        stats_table.loc[0, ['num_tokens_avg', 'num_tokens_max', 'num_tokens_min', 'num_tokens_std']] = \
            num_tokens.mean(), num_tokens.max(), num_tokens.min(), num_tokens.std()
        safe_append_dataframe_to_file(table_path=(args.tables / 'dataset_stats.csv'), table=stats_table)
