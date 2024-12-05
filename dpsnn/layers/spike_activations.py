import math

import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_activation(name):
    # print(f"activation_name: {name}")
    if name == "STDB":
        return STDBSpike.apply
    elif name == "linear":
        return LinearSpike.apply
    elif name == "gaussian":
        return GaussianSpike.apply
    elif name == "atan":
        return AtanSpike.apply
    elif name == "hard_voltage":
        return hard_voltage_transform_function.apply


class STDBSpike(torch.autograd.Function):
	alpha 	= 0.3
	beta 	= 0.01

	@staticmethod
	def forward(ctx, membrane, threshold, last_spike):
		ctx.save_for_backward(last_spike)
		out = torch.zeros_like(membrane)
		out[membrane > threshold] = 1.0
		return out

	@staticmethod
	def backward(ctx, grad_output): 
        # print(f"backward inputs num:")		
		last_spike = ctx.saved_tensors[0]
		grad_input = grad_output.clone()
		grad = STDBSpike.alpha * torch.exp(-1*last_spike)**STDBSpike.beta
		return grad * grad_input, None, None


class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, membrane, threshold, last_spike):
        ctx.save_for_backward(membrane)
        out = torch.zeros_like(membrane)
        out[membrane > threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        membrane = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad = LinearSpike.gamma * F.threshold(1.0-torch.abs(membrane), 0, 0)
        return grad * grad_input, None, None


class AtanSpike(torch.autograd.Function):
    @staticmethod
    def heaviside(x: torch.Tensor):
        return (x >= 0).to(x.dtype)

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            if isinstance(alpha, torch.Tensor):
                ctx.save_for_backward(x, alpha)
            else:
                ctx.save_for_backward(x)
                ctx.alpha = alpha
        return AtanSpike.heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        grad_alpha = None
        if ctx.saved_tensors.__len__() == 1:
            grad_x = ctx.alpha / 2 / (1 + (ctx.alpha * math.pi / 2 * ctx.saved_tensors[0]).square()) * grad_output
        else:
            # 避免重复计算，共用的部分
            shared_c = grad_output / (1 + (ctx.saved_tensors[1] * math.pi / 2 * ctx.saved_tensors[0]).square())
            if ctx.needs_input_grad[0]:
                grad_x = ctx.saved_tensors[1] / 2 * shared_c
            if ctx.needs_input_grad[1]:
                # 由于alpha只有一个元素，因此梯度需要求和，变成标量
                grad_alpha = (ctx.saved_tensors[0] / 2 * shared_c).sum()
        return grad_x, grad_alpha


class GaussianSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        scale = 6.0
        hight = 0.15
        gamma = 0.5  # gradient scale
        lens = 0.5

        temp = GaussianSpike.gaussian(input, mu=0.0, sigma=lens) * (1. + hight) \
               - GaussianSpike.gaussian(input, mu=lens, sigma=scale*lens) * hight \
               - GaussianSpike.gaussian(input, mu=-lens, sigma=scale*lens) * hight
        return grad_input * temp.float() * gamma
    
    @staticmethod
    def gaussian(x, mu=0., sigma=.5):
        return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class hard_voltage_transform_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        # v = v * (1 - spikes) + v_reset * spikes
        mask = spike.bool()  # 表示释放脉冲的位置
        if v.requires_grad and spike.requires_grad:
            ctx.save_for_backward(~mask, v_reset - v)
        elif v.requires_grad and not spike.requires_grad:
            ctx.save_for_backward(~mask)
        elif not v.requires_grad and spike.requires_grad:
            ctx.save_for_backward(v_reset - v)

        return v.masked_fill(mask, v_reset)  # 释放脉冲的位置，电压设置为v_reset，out-of-place操作

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_v = None
        grad_spike = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_v = grad_output * ctx.saved_tensors[0]
            grad_spike = grad_output * ctx.saved_tensors[1]
        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            grad_v = grad_output * ctx.saved_tensors[0]
        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_spike = grad_output * ctx.saved_tensors[0]

        return grad_v, grad_spike, None


