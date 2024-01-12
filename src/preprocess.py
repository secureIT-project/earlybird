import argparse
import collections
import itertools
import json
import logging
import pathlib
import pprint
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, OrderedDict

import jsonlines
import pandas as pd
import yaml
from more_itertools import unzip
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.utils.util import parse_args, get_time_hh_mm_ss

logger = logging.getLogger(__name__)


def remove_double_newline(line: str) -> str:
    """Ensure single line new line characters are used"""
    return re.sub(r'\n+', '\n', line).strip()


def replace_n_spaces_with_tab(line: str, n: int) -> str:
    """Ensure tabulation stands for a certain number of spaces"""
    return re.sub(r' '*n, '\t', line).strip()


def replace_separators_with_one_space(line: str) -> str:
    """Ensure single space instead of any separator"""
    return re.sub(r'\n+|\r\n|\t|\r', ' ', line).strip()


def replace_multiple_spaces_with_one_space(line: str) -> str:
    """Ensure single space instead of several in a row"""
    return re.sub(r' +', ' ', line).strip()


def process_one_python_example(line: str) -> str:
    """Replace INDENT, DEDENT, NEW_LINE with tabs and \n"""
    tokens = line.split()
    spacing = ''
    formatted_tokens = tokens.copy()
    indent_count = 0
    prev_new_line_ix = 0

    for i, t in enumerate(formatted_tokens):
        if t.lower() == 'new_line':
            formatted_tokens[i] = ' \n ' + spacing
            prev_new_line_ix = i
        elif t.lower() == 'indent':
            indent_count += 1
            spacing = ' \t ' * indent_count
            formatted_tokens[prev_new_line_ix] = ' \n ' + spacing
            formatted_tokens[i] = ''
        elif t.lower() == 'dedent':
            indent_count -= 1
            spacing = ' \t ' * indent_count
            formatted_tokens[prev_new_line_ix] = ' \n ' + spacing
            formatted_tokens[i] = ''

    output = ' '.join(formatted_tokens)
    output = re.sub(' +', ' ', output) # replace any number of spaces with one space

    return output


class Preprocess(ABC):
    """Preprocesses code classification datasets"""
    def __init__(self, args: argparse.Namespace, paths: List[pathlib.Path or str]) -> None:
        self.dataset_name = args.dataset_name
        self.splits = ['train', 'valid', 'test']
        self.args = args
        self.paths = paths

        self.data = None
        self.json_lines = dict()

        self.raw_dataset_path = Path(paths['project_path']) / paths[self.dataset_name]['raw']
        self.stats_dirpath = Path(self.paths['project_path']) / \
                             self.paths[self.dataset_name]['dataset_characteristics']
        self.stats_filepath = self.stats_dirpath / \
                              f'data_stats_{self.args.dataset_name}_' \
                              f'underrep_lim_{self.args.underrepresented_threshold}_' \
                              f'custom_split_{self.args.custom_split}_shrink_code_{self.args.shrink_code}.xlsx'

    @abstractmethod
    def read(self) -> None:
        """Read the dataset"""
        pass

    def transform(self) -> None:
        """Format the input data"""
        if self.args.shrink_code:
            self.shrink_code()

    def shrink_code(self) -> None:
        """Make code snippets more compact"""
        self.extract_code_and_labels_from_jsonlines()
        for s in self.splits:
            if args.keep_formatting:
                self.data[s]['src'] = list(map(lambda x:
                                               replace_n_spaces_with_tab(remove_double_newline(x),
                                                                         self.paths[self.dataset_name]['spaces_in_tab']),
                                               self.data[s]['src']))
            else:
                self.data[s]['src'] = list(map(lambda x:
                                               replace_multiple_spaces_with_one_space(
                                                   replace_separators_with_one_space(x)),
                                               self.data[s]['src']))
        self.update_jsonlines()

        for mode in set(self.splits) - set(['train']):
            overlapping = set([entry['src'] for entry in self.json_lines['train']]) & \
                          set([entry['src'] for entry in self.json_lines[mode]])
            logger.info(f'train - {mode} overlap: {len(overlapping)}')

    def update_jsonlines(self) -> None:
        """Create a list of dictionaries based on input data, where one dictionary stores one example"""
        if self.data:
            for s in self.splits:
                self.json_lines[s] = [{'src': code, 'label': label}
                                      for code, label in zip(self.data[s]['src'], self.data[s]['label'])]

    def ensure_stats_dirpath(self) -> None:
        """Create a folder for storing statistics about a dataset if the folder does not exist"""
        if not self.stats_dirpath.exists():
            self.stats_dirpath.mkdir(parents=True, exist_ok=True)

    def extract_code_and_labels_from_jsonlines(self) -> None:
        """For each data split (train, valid, test), extract code samples and labels in separate lists"""
        if self.data:
            return
        self.data = dict()
        for s in self.splits:
            self.data[s] = dict()
            code, labels = unzip(list(map(lambda x: (x['src'], x['label']), self.json_lines[s])))
            self.data[s]['src'], self.data[s]['label'] = list(code), list(labels)

        if not ('fixeval' in self.dataset_name) and not self.args.custom_split:
            assert set(self.data['test']['label']) == set(self.data['train']['label']) and \
                   set(self.data['test']['label']) == set(self.data['valid']['label']), \
                f"label in train, valid and test do not match:\n" \
                f"train: {set(self.data['train']['label'])}\n" \
                f"valid: {set(self.data['valid']['label'])}\n" \
                f"test: {set(self.data['test']['label'])}"

    def unique_labels(self) -> List[str]:
        """Return a list of unique labels for a given dataset"""
        self.extract_code_and_labels_from_jsonlines()
        unique_labels = sorted(list(set(self.data['test']['label']).union(set(self.data['train']['label'])). \
                                   union(set(self.data['valid']['label']))))
        return unique_labels

    def class_distribution(self) -> None:
        """Count the number of elements in each class and save statistics in an Excel file"""
        df = pd.DataFrame(columns=self.splits, index=self.unique_labels())
        for s in self.splits:
            df.loc[:, s] = pd.Series(collections.Counter(list(itertools.chain(self.data[s]['label']))))
        df = df.sort_values(by=['train', 'valid', 'test'], axis=0, ascending=False)

        self.ensure_stats_dirpath()
        with open(self.stats_dirpath / 'README.txt', 'wt') as f:
            pprint.pprint(vars(self.args), stream=f)
            pprint.pprint(self.paths, stream=f)
        logger.info(f"Preprocessing parameters are saved to {self.stats_dirpath}")

        df.loc['Total', :] = df.sum(axis=0)
        df.to_excel(self.stats_filepath, sheet_name='class_distribution')

    def token_stats(self) -> None:
        """Calculate the number of tokens in each input examples and save the statistics for each data split to Excel"""
        logger.info('Tokenization'); self.extract_code_and_labels_from_jsonlines()
        tokenizer = AutoTokenizer.from_pretrained(Path(self.paths['project_path']) / self.args.tokenizer_path)

        df_high_level = None
        df_class_level = None

        for s in self.splits:
            logger.info(f'Split {s}')
            for i in range(3):
                code = self.data[s]['src'][i]
                tokens = tokenizer.tokenize(code)
                logger.info(f'\nCode example {i}\n\n{code}\n\nTokens\n\n{tokens}\n\nToken length: \t {len(tokens)}')
            # Count tokens for one split
            src_tokenized_len = [len(tokenizer.tokenize(code)) for code in self.data[s]['src']]
            # Combine counts in a table
            df_aux = pd.DataFrame({s: src_tokenized_len, 'label': self.data[s]['label']})
            # Append to the dataset overview table
            df_high_level = df_aux.describe() if df_high_level is None \
                else pd.concat([df_high_level, df_aux.describe()], axis='columns')
            # Append to the dataset overview by class
            df_aux_class_level = df_aux.groupby('label').describe()
            df_class_level = df_aux_class_level.sort_values(by=[(s, 'count')], ascending=False) \
                if df_class_level is None \
                else pd.concat([df_class_level, df_aux_class_level], axis='columns')
            logger.info(f'Tokenized {s}')
        logger.info(f'Tokenized all')

        self.ensure_stats_dirpath()
        with pd.ExcelWriter(self.stats_filepath, mode='a') as writer:
            df_high_level.round().to_excel(writer, sheet_name='token_count')
            df_class_level.round().to_excel(writer, sheet_name='token_count_per_class')

    @abstractmethod
    def save(self) -> None:
        """Save the formatted dataset"""
        pass

    def ensure_data_output_dir(self) -> None:
        """Ensure the folder for storing the formatted data exists"""
        dirpath = (Path(self.paths['project_path']) / self.paths[self.dataset_name]['processed']['train_data_path']).parent
        if not dirpath.exists():
            dirpath.mkdir(parents=True, exist_ok=True)

    def save_jsonl(self) -> None:
        """Save the (formatted) list of dictionaries as jsonlines"""
        self.ensure_data_output_dir()
        for s in self.splits:
            filename = Path(self.paths['project_path']) / self.paths[self.dataset_name]['processed'][f'{s}_data_path']
            with jsonlines.open(filename, 'w') as writer:
                writer.write_all(self.json_lines[s])


class PreprocessCubert(Preprocess, ABC):
    def __init__(self, args, paths):
        super().__init__(args, paths)
        self.input_splits = ['train', 'dev', 'eval']
        self.n_subsets = self.paths[self.dataset_name]['n_subsets']

    def read(self):
        # read all data in the loop over splits (dev, train, test) and existing files
        for s_in, s_out in zip(self.input_splits, self.splits):
            assert s_in == 'dev' and s_out == 'valid' or \
                   s_in == 'train' and s_out == 'train' or \
                   s_in == 'eval' and s_out == 'test', f'Read split {s_in} does not match write split {s_out}'

            self.json_lines[s_out] = []

            for subset in range(self.n_subsets):
                logger.info(f'Read {s_in} subset {subset} out of {self.n_subsets}')
                filepath_json_dump = self.raw_dataset_path / \
                                     f'20200621_Python_{self.dataset_name}_datasets_{s_in}.' \
                                     f'jsontxt-0000{subset}-of-0000{self.n_subsets}'

                with open(filepath_json_dump, 'r') as f:
                    data_jsonl = list(f)

                def process_json_line(l):
                    raw_line = json.loads(l)
                    return {'src': raw_line['function'], 'label': raw_line['label']}

                self.json_lines[s_out] += list(map(lambda x: process_json_line(x), data_jsonl))
                logger.info(f'N samples: {len(data_jsonl)}')

            logger.info(f'{s_out} contains {len(self.json_lines[s_out])} jsonlines')

    def save(self):
        self.save_jsonl()


class PreprocessDevign(Preprocess, ABC):
    """Code is mostly taken from
    https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Defect-detection/dataset/preprocess.py
    """

    def __init__(self, args, paths):
        super().__init__(args, paths)
        self.js_all = None
        self.indexes = dict()

    def read(self):
        self.js_all = json.load(open(self.raw_dataset_path / 'function.json'))
        for s in self.splits:
            self.indexes[s] = set()
            with open(self.raw_dataset_path / f'{s}.txt') as f:
                for line in f:
                    line = line.strip()
                    self.indexes[s].add(int(line))

    def transform(self):
        self.data = dict()
        for s in self.splits:
            self.data[s] = dict()
            self.data[s]['src'] = []
            self.data[s]['label'] = []
            self.json_lines[s] = []
            for idx, js in enumerate(self.js_all):
                if idx in self.indexes[s]:
                    code = ' '.join(js['func'].split()) if self.args.shrink_code else js['func']
                    self.data[s]['src'].append(code)
                    self.data[s]['label'].append(js['target'])
                    self.json_lines[s] += [{'idx': idx, 'src': code, 'label': js['target']}]

    def save(self):
        self.save_jsonl()


class PreprocessBreakItFixIt(Preprocess, ABC):
    def __init__(self, args, paths):
        super().__init__(args, paths)
        self.js_all = None
        self.indexes = dict()
        self.label_mapping = {'unbalanced (){}[]': 'unbalanced (){}[]',
                              'expected an indented block': 'indentation error',
                              'unexpected indent': 'indentation error',
                              'unexpected unindent': 'indentation error',
                              'unindent does not match any outer indentation level': 'indentation error',
                              'invalid syntax': 'invalid syntax'}

    def read(self):
        self.js_all = json.load(open(self.raw_dataset_path / 'orig.bad.json'), object_pairs_hook=OrderedDict)
        example_keys = list(self.js_all.keys())
        csize = len(example_keys) // 5 + 1

        # read train
        json_lines_train = []
        for cluster in [0, 1, 2]:
            for _id in example_keys[csize * cluster: csize * (cluster + 1)]:
                example = self.js_all[_id]
                json_lines_train.append({
                    'id': _id,
                    'src': example['code_string'],
                    'label': self.label_mapping[example['err_obj']['msg']]
                })

        # split train into train and validation, NB cmd line args --ratio_train 0.9
        train_df = pd.DataFrame(json_lines_train)

        ix = {}
        ix['train'], ix['valid'], _, _ = train_test_split(
            train_df.id, train_df.label, test_size=1. - self.args.ratio_train,
            random_state=self.args.seed, shuffle=True, stratify=train_df.label)

        del train_df

        self.json_lines['train'] = []
        self.json_lines['valid'] = []
        for i in range(len(json_lines_train)):
            example = json_lines_train[i]
            self.json_lines['train'].append(example) \
                if example['id'] in list(ix['train']) else self.json_lines['valid'].append(example)

        # read test
        self.json_lines['test'] = []
        for cluster in [3, 4]:
            for _id in example_keys[csize * cluster: csize * (cluster + 1)]:
                example = self.js_all[_id]
                self.json_lines['test'].append({
                    'id': _id,
                    'src': example['code_string'],
                    'label': self.label_mapping[example['err_obj']['msg']]
                })

        for s in self.splits:
            logger.info(f'{s} contains {len(self.json_lines[s])} jsonlines')

    def save(self):
        self.save_jsonl()


class PreprocessReveal(Preprocess, ABC):
    def __init__(self, args, paths):
        super().__init__(args, paths)
        self.js_all = []

    def read(self):
        js_vulnerable = json.load(open(self.raw_dataset_path / 'vulnerables.json'), object_pairs_hook=OrderedDict)
        js_non_vulnerable = json.load(open(self.raw_dataset_path / 'non-vulnerables.json'), object_pairs_hook=OrderedDict)

        # rename keys in dicts and add labels
        for example in js_vulnerable:
            self.js_all.append({
                'src': example['code'],
                'label': 'vulnerable'
            })
        for example in js_non_vulnerable:
            self.js_all.append({
                'src': example['code'],
                'label': 'non-vulnerable'
            })
        logging.info(f'# vulnerable: \t {len(js_vulnerable)}\n'
                     f'# non-vulnerable: \t {len(js_non_vulnerable)}')

    def transform(self):
        # split train into train and validation, NB cmd line args --ratio_train 0.9
        df = pd.DataFrame(self.js_all)

        ix = dict()
        label = dict()
        indexes = df.index.values

        ix['train'], ix_test_aux, label['train'], label_test_aux = train_test_split(
            indexes, df.label, test_size=1. - self.args.ratio_train,
            random_state=self.args.seed, shuffle=True, stratify=df.label)

        ix['valid'], ix['test'], label['valid'], label['test'] = train_test_split(
            ix_test_aux, label_test_aux,
            test_size=self.args.ratio_test / (self.args.ratio_test + self.args.ratio_valid),
            random_state=self.args.seed, shuffle=True, stratify=label_test_aux)

        for s in self.splits:
            self.json_lines[s] = []
            for idx, l in zip(ix[s], label[s]):
                self.json_lines[s].append({
                    'src': df.loc[idx, 'src'],
                    'label': df.loc[idx, 'label']
                })
        self.js_all = None

        super().transform()

    def save(self):
        self.save_jsonl()


if __name__ == '__main__':
    # Track time
    time_start = time.time()

    # Parse command line and yaml arguments
    args = parse_args(sys.argv[1:])
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Logger
    logdir = Path(config['project_path']) / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    logfilename = '_'.join(['preprocess', args.dataset_name, 'shrink_code', str(args.shrink_code)]
                           if not 'fixeval' in args.dataset_name
                           else
                           ['preprocess', args.dataset_name,
                            'underrep_lim', str(args.underrepresented_threshold),
                            'custom_split', str(args.custom_split),
                            'shrink_code', str(args.shrink_code)
                            ]) + '.log'
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO,
                        filename=logdir / logfilename)

    modes = ['train', 'valid', 'test']

    # Preprocess
    # choose preprocessing function based on dataset name
    dataset = {
        'exception': PreprocessCubert,
        'swapped_operands': PreprocessCubert,
        'wrong_binary_operator': PreprocessCubert,
        'variable_misuse': PreprocessCubert,
        'devign': PreprocessDevign,
        'break_it_fix_it': PreprocessBreakItFixIt,
        'reveal': PreprocessReveal
    }[args.dataset_name](args, config)

    dataset.read()
    dataset.transform()
    dataset.save()
    dataset.class_distribution()
    dataset.token_stats()

    time_stop = time.time()
    get_time_hh_mm_ss(int(time_stop - time_start))

    logging.info(f"End of execution.")

