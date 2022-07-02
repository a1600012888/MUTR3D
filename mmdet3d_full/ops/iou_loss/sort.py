import torch
from torch import nn
from torch.autograd import Function
from .iou_loss_ext import sort_vertices_forward


class SortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        idx = sort_vertices_forward(vertices, mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx
    
    @staticmethod
    def backward(ctx, gradout):
        return ()

sort_v = SortVertices.apply