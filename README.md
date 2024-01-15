[![source under MIT licence](https://img.shields.io/badge/source%20license-MIT-green)](LICENSE.txt)
[![data under CC BY 4.0 license](https://img.shields.io/badge/data%20license-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)
<a href="https://doi.org/10.5281/zenodo.7608802">
  <img align="right" src="https://zenodo.org/badge/DOI/10.5281/zenodo.7608802.svg" alt="DOI: 10.5281/zenodo.7608802">
 </a>
 
 
# Replication Package for _"The EarlyBIRD Catches the Bug: On Exploiting Early Layers of Encoder Models for More Efficient Code Classification"_

This repository contains the replication package for the paper "The EarlyBIRD Catches the Bug: On Exploiting Early Layers of Encoder Models for More Efficient Code Classification" by Anastasiia Grishina, Max Hort and Leon Moonen, published in the ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE 2023).

The paper is deposited on [arXiv](https://arxiv.org/abs/2305.04940), available under open access at the publisher's site ([ACM](https://dl.acm.org/doi/abs/10.1145/3611643.3616304)), and a copy is included in this repository.

The replication package is archived on Zenodo with DOI: [10.5281/zenodo.7608802](https://doi.org/10.5281/zenodo.7608802). 
The source code is distributed under the MIT license, the data is distributed under the CC BY 4.0 license. 


## Organization

The overall organization of the current repository is as follows:

* [src](./src) - the source code

* [requirements](./requirements) - txt files with Python packages and versions for replication

* [data](./data) - all raw datasets used for training (to be found in Zenodo [10.5281/zenodo.7608802](https://doi.org/10.5281/zenodo.7608802))

* [notebooks](./notebooks) - one Jupyter notebook with code to generate figures and tables with aggregated results as reported in the paper  

* [mlruns](./mlruns) - results of experiments, the folder is created once the run.py is executed (see part II)

* [output](./output) - results of experiments, the folder and subfodlers are created once the run.py is executed 
  * tables
  * figures
  * runs           - folder to store model checkpoints, if the corresponding command line argument (`--save_model`) is provided when running the code



## Usage

Python version: `3.7.9` (later versions should also work well); CUDA version: `11.6`; Git LFS.

Commands below work well on Mac or Linux and should be adapted if you have a Windows machine. 

### I. Set up data, environment and code

#### 1. Path to project directory

Update path/to/project to point at EarlyBIRD

```
export EarlyBIRD=~/path/to/EarlyBIRD
```

#### 2. Download codebert checkpoint
Please, install Git LFS: <https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage>

Run the following from within `$EarlyBIRD/`:

```
cd $EarlyBIRD
mkdir -p checkpoints/reused/model
cd checkpoints/reused/model
git lfs install
git clone https://huggingface.co/microsoft/codebert-base
cd codebert-base/
git lfs pull
cd ../../..
``` 

#### 3. Set up a virtual environment

```
cd $EarlyBIRD
python -m venv venv
source venv/bin/activate
```

##### 3.1 No CUDA

```
python -m pip install -r requirements/requirements_no_cuda.txt
```

##### 3.2 With CUDA (to run on GPU)

```
python -m pip install -r requirements/requirements_with_cuda.txt
python -m pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```


#### 4 Preprocess data

After preprocessing, all datasets are stored in jsonlines (if in python) format.
Naming convention: split is one of `'train', 'valid', 'test'` in 
`data/preprocessed-final/<dataset_name>/<split>.jsonl`, with 

```
{'src': "def function_1() ...", 'label': "Label1"}
{'src': "def function_2() ...", 'label': "Label2"}
...
```


##### 4.1 Devign 

Raw data is downloaded from <https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view>. 
Test, train, valid txt files are downloaded from <https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection/dataset>. 
All files are saved in `data/raw/devign`.

To preprocess raw data and save tokenization statistics with the specified tokenizer:

```
cd $EarlyBIRD
python -m src.preprocess \
    --dataset_name devign \
    --shrink_code \
    --config_path src/config.yaml \
    --tokenizer_path "checkpoints/reused/model/codebert-base"
```

##### 4.2 ReVeal

Raw data is downloaded from <https://github.com/VulDetProject/ReVeal> under 
"Our Collected vulnerabilities from Chrome and Debian issue trackers (Often referred as Chrome+Debian or Verum dataset in this project)" and saved in `data/raw/reveal`.

To preprocess raw data:

```
cd $EarlyBIRD
python -m src.preprocess \
    --dataset_name reveal \
    --shrink_code \
    --config_path src/config.yaml \
    --tokenizer_path "checkpoints/reused/model/codebert-base"
```

##### 4.3 Break-it-fix-it

Raw data is downloaded as `data_minimal.zip` from <https://github.com/michiyasunaga/BIFI> under p. 1, 
unzipped, and the folder `orig_bad_code` is saved in `data/raw/break_it_fix_it`.

To preprocess raw data:

```
cd $EarlyBIRD
python -m src.preprocess \
    --dataset_name break_it_fix_it \
    --shrink_code \
    --ratio_train 0.9 \
    --config_path src/config.yaml \
    --tokenizer_path "checkpoints/reused/model/codebert-base"
```

Note:
The original paper contains only train and test split. 
Use `--ratio_train` to specify what part of the original train (orig-train) split 
will be used in train and the rest of orig-train will be used for validation during training.

##### 4.4 Exception Type

Raw data is downloaded from <https://github.com/google-research/google-research/tree/master/cubert> under "2. Exception classification" 
(it points to [this storage](https://console.cloud.google.com/storage/browser/cubert/20200621_Python/exception_datasets;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)) 
and saved in `data/raw/exception_type`.

To preprocess raw data:

```
cd $EarlyBIRD
python -m src.preprocess \
    --dataset_name exception \
    --shrink_code \
    --config_path src/config.yaml \
    --tokenizer_path "checkpoints/reused/model/codebert-base"
```


### II. Run code

Activate virtual environment (if not done so yet):

```
cd $EarlyBIRD
source venv/bin/activate
```

#### Example run

Run experiments with Devign using pruned models (`cutoff_layers_one_layer_cls`) to 3 layers (`--hidden_layer_to_use 3`), for example:

```
cd $EarlyBIRD
python -m src.run --help              # for help with command line args

python -m src.run \
    --config_path src/config.yaml \
    --model_name codebert \
    --model_path "checkpoints/reused/model/codebert-base" \
    --tokenizer_path "checkpoints/reused/model/codebert-base" \
    --dataset_name devign \
    --benchmark_name acc \
    --train \
    --test \
    -warmup 0 \
    --device cuda \
    --epochs 10 \
    -clf one_linear_layer \
    --combination_type cutoff_layers_one_layer_cls \
    --hidden_layer_to_use 3 \
    --experiment_no 12 \
    --seed 42
```

To run experiments on a small subset of data, use `--debug` argument and reduce the number of epochs. For example:

```
python -m src.run \
    --debug \
    --config_path src/config.yaml \
    --model_name codebert \
    --model_path "checkpoints/reused/model/codebert-base" \
    --tokenizer_path "checkpoints/reused/model/codebert-base" \
    --dataset_name devign \
    --benchmark_name acc \
    --train \
    --test \
    -warmup 0 \
    --device cuda \
    --epochs 2 \
    -clf one_linear_layer \
    --combination_type cutoff_layers_one_layer_cls \
    --hidden_layer_to_use 3 \
    --experiment_no 12 \
    --seed 42
```

## Explore output

Your `EarlyBIRD/` should contain `mlruns/`. 
If you started the `run.py` from another location, 
you will find `mlruns/` one level below that location.

```
cd $EarlyBIRD
mlflow ui
```


## Citation

If you build on this data or code, please cite this work by referring to the paper:

```
@inproceedings{grishina2023:earlybird,
   title = {The EarlyBIRD Catches the Bug: On Exploiting Early Layers of Encoder Models for More Efficient Code Classification},
   author = {Anastasiia Grishina and Max Hort and Leon Moonen},
   booktitle = {ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE)},
   year = {2023},
   publisher = {ACM},
   doi = {https://doi.org/10.1145/3611643.3616304},
   note = {Pre-print on arXiv at https://arxiv.org/abs/2305.04940}
}
```

## Changelog
- v1.0 - initial conference submission package: raw data, code, and example outputs. 
The config files are adapted to run only CodeBert experiments.
- v1.1 - added: model checkpoints with the best improving combinations, jupyter notebook to obtain figures in the paper; 
updated: config files and code to specify and run an arbitrary encoder model;
- v1.2 - updated code with documentation and typing hints; added a link to the public GitHub repository to README. 

## Acknowledgement

The work included in this repository was supported by the Research Council of Norway through the secureIT project (IKTPLUSS #288787). Max Hort is supported through the ERCIM ‘Alain Bensoussan’ Fellowship Programme. The empirical evaluation was performed on the Experimental Infrastructure for Exploration of Exascale Computing (eX3), financially supported by the Research Council of Norway under contract #270053, as well as on resources provided by Sigma2, the National Infrastructure for High Performance Computing and Data Storage in Norway.
