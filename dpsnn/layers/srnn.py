from joblib import Parallel, delayed
import numpy as np

from pesq import pesq_batch, PesqError

import torch as th
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
import torchaudio

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from ..layers import accelerating, surrogate
from .sdr import pairwise_neg_sisdr, singlesrc_neg_sisdr
from .spike_neurons import PLIFNode, ALIFNode
from .sequential import Sequential
from .spike_neurons import get_neuro


class SRNN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 lif_type, lif_config, bias=True):
        super(SRNN, self).__init__()
        
        self.neuro = get_neuro(lif_type,  **{**lif_config, "input_dim":output_dim})

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
        self.recurrent = nn.Linear(output_dim, output_dim, bias=bias)

    def forward(self, x, step, detach=False):
        if detach:
            self.neuro.detach()

        x = th.reshape(x, (-1, self.input_dim))
        y_dense = self.dense(x)

        # print(f"input shape1: {x.shape}")
        spike, _ = self.neuro.get_neuro_states(y_dense, step)
        # print(f"spike neuro shape1: {spike.shape}")
        spike = th.reshape(spike, (-1, self.output_dim))
        # print(f"spike neuro shape2: {spike.shape}")
        
        y_recurrent = self.recurrent(spike)
        y = y_dense + y_recurrent
        # print(f"y shape: {y.shape}")
        y, _ = self.neuro(y, step)
        # print(f"y spike shape: {y.shape}")
        return y


class ReadoutLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 lif_type,
                 lif_config,
                 bias=True):
        super(ReadoutLayer, self).__init__()
        
        # if non_linear not in supported_nonlinear:
        #     raise RuntimeError("Unsupported non-linear function: {}",
        #                        format(non_linear))
        # self.non_linear_type = non_linear
        lif_config["input_dim"] = output_dim
        self.neuro = get_neuro(lif_type, **{**lif_config, "input_dim":output_dim, "no_spiking":True})

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
        # self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x, step, detach=False):
        if detach:
            self.neuro.detach()

        # print(f"x neuro shape: {x.shape}")
        x = th.reshape(x, (-1, self.input_dim))
        y = self.dense(x)
        # y = self.batch_norm(y)
        
        _, mem = self.neuro(y, step)
        # print(f"y spike shape: {y.shape}")
        # print(f"output spike shape: {y.shape}")
        return mem