import os
import random
from pathlib import Path

import torch
import math
import torch.nn.functional as F

from .salo_spattn import matchingStatic_Block


def _eval_overall_sparsity(sparsity_mask, attn_mask):

    attn_mask = (attn_mask > -1).float()
    attn_mask = attn_mask * (attn_mask.permute(0, 1, 3, 2))

    length = attn_mask.shape[-1]
    attn_mask = attn_mask * torch.tril(torch.ones((length, length)), diagonal=0).cuda()

    scaling_factor = attn_mask.mean(dim=(1, 2, 3))
    sparsity_per_seq = (sparsity_mask.float() * attn_mask).mean(dim=(1, 2, 3))
    overall_sparsity = (sparsity_per_seq / scaling_factor).mean().item()
    return overall_sparsity


count = 0
mean_len = 0
all_sparisity = 0


def gen_sparsity_mask_xm(attention_scores, attn_mask, threshold_0, threshold_1):
    global count
    global mean_len
    global all_sparisity

    attention_scores = F.softmax(attention_scores + attn_mask, dim=-1)

    n1 = 16
    n2 = 8
    m = 64

    original_shape = attention_scores.shape
    token_len = original_shape[-1]
    s = token_len // m
    reshaped_scores = attention_scores.view(*original_shape[:-1], s, m)
    sum_m = torch.sum(reshaped_scores, dim=-1, keepdim=True).expand_as(reshaped_scores)
    sum_m = sum_m * token_len / m
    _, indices1 = torch.topk(reshaped_scores, n1, dim=-1, largest=True)
    _, indices2 = torch.topk(reshaped_scores, n2, dim=-1, largest=True)
    sparsity_mask_reshaped = torch.zeros_like(reshaped_scores, dtype=torch.bool)
    sparsity_mask_reshaped1 = torch.zeros_like(reshaped_scores, dtype=torch.bool).scatter_(-1, indices1, True)
    sparsity_mask_reshaped2 = torch.zeros_like(reshaped_scores, dtype=torch.bool).scatter_(-1, indices2, True)
    sparsity_mask_reshaped = torch.where(sum_m < threshold_1, sparsity_mask_reshaped, sparsity_mask_reshaped2)
    sparsity_mask_reshaped = torch.where(sum_m > threshold_0, sparsity_mask_reshaped1, sparsity_mask_reshaped)
    sparsity_mask = sparsity_mask_reshaped.view(original_shape)


    count += 1
    all_sparisity += _eval_overall_sparsity(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(count == 1) :
        with open('sparsity_xm.txt', 'w') as txt:
            txt.writelines('eval_xm:\n')
    if(count%100 == 0) :
        with open('sparsity_xm.txt', 'a') as txt:
            txt.writelines('mean_sparisity: {:<10.5f} mean_len: {:<10} count: {:<10}\n'
                           .format(round(all_sparisity/count, 5), round(mean_len.item()/count, 1), count))
       
    sparsity_mask = sparsity_mask.type_as(attention_scores)
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0

    return sparsity_mask.detach()


def gen_sparsity_mask_nm(attention_scores, attn_mask, m, n):
    global count
    global mean_len
    global all_sparisity

    attention_scores = F.softmax(attention_scores + attn_mask, dim=-1)

    original_shape = attention_scores.shape
    token_len = original_shape[-1]
    s = token_len // m
    reshaped_scores = attention_scores.view(*original_shape[:-1], s, m)
    _, indices = torch.topk(reshaped_scores, n, dim=-1, largest=True)
    sparsity_mask_reshaped = torch.zeros_like(reshaped_scores, dtype=torch.bool)
    sparsity_mask_reshaped.scatter_(-1, indices, True)
    sparsity_mask = sparsity_mask_reshaped.view(original_shape)

    count += 1
    all_sparisity += _eval_overall_sparsity(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(count == 1) :
        with open('sparsity_nm.txt', 'w') as txt:
            txt.writelines('eval_nm:\n')
    if(count%100 == 0) :
        with open('sparsity_nm.txt', 'a') as txt:
            txt.writelines('mean_sparisity: {:<10.5f} mean_len: {:<10} count: {:<10}\n'
                           .format(round(all_sparisity/count, 5), round(mean_len.item()/count, 1), count))

    
    sparsity_mask = sparsity_mask.type_as(attention_scores)
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    
    return sparsity_mask.detach()


def gen_sparsity_mask_sanger(attention_scores, attn_mask, threshold):
    global count
    global mean_len
    global all_sparisity

    attention_scores = F.softmax(attention_scores + attn_mask, dim=-1)
    sparsity_mask = attention_scores > threshold

    count += 1
    all_sparisity += _eval_overall_sparsity(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(count == 1) :
        with open('sparsity_sanger.txt', 'w') as txt:
            txt.writelines('eval_sanger:\n')
    if(count%100 == 0) :
        with open('sparsity_sanger.txt', 'a') as txt:
            txt.writelines('mean_sparisity: {:<10.5f} mean_len: {:<10} count: {:<10}\n'
                           .format(round(all_sparisity/count, 5), round(mean_len.item()/count, 1), count))

    sparsity_mask = sparsity_mask.type_as(attention_scores)
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    
    return sparsity_mask.detach()


def gen_sparsity_mask_topk(attention_scores, attn_mask, topk):
    global count
    global mean_len
    global all_sparisity

    attention_scores = F.softmax(attention_scores + attn_mask, dim=-1)
    sparsity_mask = torch.full_like(attention_scores, False, dtype=torch.bool)
    index = torch.topk(attention_scores, topk, dim=-1, largest=True)[1]
    sparsity_mask.scatter_(-1, index, True)

    count += 1
    all_sparisity += _eval_overall_sparsity(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(count == 1) :
        with open('sparsity_topk.txt', 'w') as txt:
            txt.writelines('eval_topk:\n')
    if(count%100 == 0) :
        with open('sparsity_topk.txt', 'a') as txt:
            txt.writelines('mean_sparisity: {:<10.5f} mean_len: {:<10} count: {:<10}\n'
                           .format(round(all_sparisity/count, 5), round(mean_len.item()/count, 1), count))
    
    sparsity_mask = sparsity_mask.type_as(attention_scores)
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0

    return sparsity_mask.detach()


def gen_sparsity_mask_salo(attention_scores, attn_mask):
    global count
    global mean_len
    global all_sparisity

    match_size = 64
    pe_size = 8 
    global_nums = 1 
    random_nums = 3
    dilation = 1

    attention_scores_salo = attention_scores
    attn_mask_salo = attn_mask
    
    sparsity_mask = matchingStatic_Block(attention_scores_salo, attn_mask_salo, match_size, pe_size, global_nums, random_nums, dilation)

    count += 1
    all_sparisity += _eval_overall_sparsity(sparsity_mask, attn_mask)
    mean_len += torch.mean(torch.sum((attn_mask > -1).float(), dim=-1))
    if(count == 1) :
        with open('sparsity_salo.txt', 'w') as txt:
            txt.writelines('eval_salo:\n')
    if(count%100 == 0) :
        with open('sparsity_salo.txt', 'a') as txt:
            txt.writelines('mean_sparisity: {:<10.5f} mean_len: {:<10} count: {:<10}\n'
                           .format(round(all_sparisity/count, 5), round(mean_len.item()/count, 1), count))

    
    sparsity_mask = sparsity_mask.type_as(attention_scores)
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    
    return sparsity_mask.detach()


def prune_attn_scores(attn_scores, attn_mask, threshold_0 = 1.0, threshold_1 = 0.1, m=64, n=16, topk=100, threshold=1e-4, sparse_method="xm"):
    match sparse_method:
        case "xm":
            return gen_sparsity_mask_xm(attn_scores, attn_mask, threshold_0, threshold_1)
        case "nm":
            return gen_sparsity_mask_nm(attn_scores, attn_mask, m, n)
        case "sanger":
            return gen_sparsity_mask_sanger(attn_scores, attn_mask, threshold)
        case "topk":
            return gen_sparsity_mask_topk(attn_scores, attn_mask, topk)
        case "salo":
            return gen_sparsity_mask_salo(attn_scores, attn_mask)
        case _:
            print("Please set sparse_method: xm, nm, sanger, salo or topk")
            exit()


def quant_qk_matmul(quant_methed, query_layer, key_layer, quant_matmul=None):

    if(quant_methed == "1_2_4bit"):
        last_dim = query_layer.shape[-1]
        assert last_dim % 2 == 0, "last_dim must even"
        sparse_query_layer = torch.zeros_like(query_layer)
        for i in range(0, last_dim, 2):
            part = query_layer[..., i:i+2]
            abs_part = torch.abs(part)
            max_index = torch.argmax(abs_part, dim=-1, keepdim=True)
            sparse_part = torch.zeros_like(part).scatter_(-1, max_index, part.gather(-1, max_index))
            sparse_query_layer[..., i:i+2] = sparse_part
        query_layer = sparse_query_layer

    elif(quant_methed == "1_4_6bit"):
        last_dim = query_layer.shape[-1]
        assert last_dim % 4 == 0, "last_dim must be divisible by 4"
        sparse_query_layer = torch.zeros_like(query_layer)
        for i in range(0, last_dim, 4):
            part = query_layer[..., i:i+4]
            abs_part = torch.abs(part)
            max_index = torch.argmax(abs_part, dim=-1, keepdim=True)
            sparse_part = torch.zeros_like(part).scatter_(-1, max_index, part.gather(-1, max_index))
            sparse_query_layer[..., i:i+4] = sparse_part
        query_layer = sparse_query_layer

    quant_attention_scores = quant_matmul(query_layer, key_layer)

    return quant_attention_scores