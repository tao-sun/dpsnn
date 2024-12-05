# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import torch as th
from torch import nn
# from torch.nn import functional as F

# from . import dsp

def remix(noisy, clean):
    """
    Mixes different noises with clean speech within a given batch
    """
    noise = noisy - clean
    bs, *other = noise.shape
    perm = th.argsort(th.rand(bs), dim=0)
    noisy = noise[perm]+clean
    return noisy, clean


def remix_split(noisy, clean, split=64):
    """
    Mixes different noises with clean speech within a given batch
    """
    noise = noisy - clean
    bs, *other = noise.shape
    device = noise.device

    perms = []
    split_perm = th.argsort(th.rand(split), dim=0)
    perms.append(split_perm)
    for i in range(bs, 64):
        split_perm += 64
        perms.append(split_perm)
    rear_perm = th.argsort(th.rand(bs%split), dim=0)
    perms.append(rear_perm)
    
    perm = th.concat(perms)
    noisy = noise[perm] + clean
    return noisy, clean
