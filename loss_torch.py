import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)

def bpr_loss_pop(user_emb, pos_item_emb, neg_item_emb, pos_ipop, neg_ipop, alpha):
    # 计算正样本和负样本的得分
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)

    # 计算物品流行度的影响s
    ipop_diff = pos_ipop - neg_ipop
    weights = torch.exp(-alpha * ipop_diff)  # 计算每个样本对的权重

    # 计算BPR损失
    base_loss = -torch.log(1e-6 + torch.sigmoid(pos_score - neg_score))
    weighted_loss = weights * base_loss  # 应用权重

    return torch.mean(weighted_loss)

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg


def contrastLoss_ln_var(embeds,temp,high_freq_ratio, noise_scale, noise_type):
    anchor_embeds = embeds
    power_embeds= generate_contrastive_embedding(anchor_embeds,high_freq_ratio, noise_scale, noise_type)

    anchor_embeds_norm = F.normalize(anchor_embeds + 1e-8, p=2)
    power_embeds_norm = F.normalize(power_embeds + 1e-8, p=2)

    nume = torch.exp(torch.sum(anchor_embeds_norm * power_embeds_norm, dim=-1) / temp)
    deno = torch.exp(anchor_embeds_norm @ power_embeds_norm.T / temp).sum(-1) + 1e-8

    sslloss = -torch.log(nume / deno).mean()

    return sslloss

def generate_contrastive_embedding(E,high_freq_ratio=0.4, noise_scale=0.01, noise_type='uniform', scaling_factor=None):
    # 1. Calculate variance for each feature dimension
    variances = torch.var(E, dim=0)

    # 2. Determine high-frequency dimensions
    num_high_freq_dims = int(high_freq_ratio * E.shape[1])
    _, high_freq_indices = torch.topk(variances, num_high_freq_dims)

    # 3. Create a contrastive embedding VE by adding noise to high-frequency dimensions
    VE = E.clone()
    if noise_type == 'normal':
        noise = torch.normal(0, noise_scale, size=(E.shape[0], num_high_freq_dims)).to(E.device)
    elif noise_type == 'uniform':
        noise = torch.rand(E.shape[0], num_high_freq_dims).to(E.device) * noise_scale
    elif noise_type == 'simgcl':
        random_noise = torch.rand_like(VE[:, high_freq_indices]).cuda()
        # noise = torch.sign(VE[:, high_freq_indices]) * F.normalize(random_noise, dim=-1) * eps
        noise = F.normalize(random_noise, dim=-1) * noise_scale
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    VE[:, high_freq_indices] += noise

    # 4. Optionally, scale the high-frequency features
    if scaling_factor:
        VE[:, high_freq_indices] *= scaling_factor

    return VE
