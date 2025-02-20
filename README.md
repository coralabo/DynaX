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

## Datasets Installation

# Experiments

## Evaluation Models

## Fine-tune Models

# Citation
Xiao Xiong, Zhaorui Chen, Yue Liang, Minghao Tian, Jiaxing Shang, Jiang Zhong, Dajiang Liu. DynaX: Sparse Attention Acceleration with Dynamic X:M Fine-Grained Structured Pruning. The ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS’25), 2025.
