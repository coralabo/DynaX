# DynaX (Updating...)
Artifact Evaluation Reproduction for "DynaX: Sparse Attention Acceleration with Dynamic X:M Fine-Grained Structured Pruning", ASPLOS 2025

## Table of contens
1. [Directory Structure](#directory-structure)
2. [Getting Started](#getting-started)
3. [Experiments](#experiments)
4. [Citation](#citation)

# Directory Structure
```
DynaX     
|  README.md     
|  dataUtil.py  
|  eval_bloom.py 
|  eval_llama3.py 
|  qlora_merge.py
|  train_bloom.py 
|  train_llama.py
|--datasets
|--newModels     
|--output
|--configs
|   |  config.json
|   |  dense_config.json
|   |  nm_sparse_config.json
|   |  salo_sparse_config.json
|   |  sanger_sparse_config.json
|   |  xm_sparse_config.json
|   |  xm_sparse_config2.json 
|--models
|   |--utils
|   |   |  quant_utils.py
|   |   |  salo_spattn.py
|   |   |  sparse_attention.py
|   |  bloom_modeling.py
|   |  llama_modeling.py       
```

# Getting Started

## Requirements
python >= 3.10<br>
torch >= 2.3.0<br>
cuda >= 12.1<br>
datasets >= 3.2.0<br>
transformers >= 4.48.3<br>
peft >= 0.14.0

## Datasets Installation
By loading the dataset online or downloading it through the following link<br>
Download the wikitext2 dataset from [here](https://huggingface.co/datasets/mindchain/wikitext2)<br>
Download the ptb dataset from [here](https://huggingface.co/datasets/ptb-text-only/ptb_text_only)<br>
Download the c4 dataset from [here]([www.baidu.com](https://huggingface.co/datasets/allenai/c4)

# Experiments

## Evaluation Models

## Fine-tune Models

# Citation
Xiao Xiong, Zhaorui Chen, Yue Liang, Minghao Tian, Jiaxing Shang, Jiang Zhong, Dajiang Liu. DynaX: Sparse Attention Acceleration with Dynamic X:M Fine-Grained Structured Pruning. The ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS’25), 2025.
