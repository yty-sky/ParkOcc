# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn

from ocnn.octree import Octree
from ocnn.utils import scatter_add


OctreeBatchNorm = torch.nn.BatchNorm1d


class OctreeGroupNorm(torch.nn.Module):
  r''' An group normalization layer for the octree.
  '''

  def __init__(self, in_channels: int, group: int, nempty: bool = False):
    super().__init__()
    self.eps = 1e-5
    self.nempty = nempty
    self.group = group
    self.in_channels = in_channels

    assert in_channels % group == 0
    self.channels_per_group = in_channels // group

    self.weights = torch.nn.Parameter(torch.Tensor(1, in_channels))
    self.bias = torch.nn.Parameter(torch.Tensor(1, in_channels))
    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.ones_(self.weights)
    torch.nn.init.zeros_(self.bias)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    batch_size = 1
    # batch_id = octree.batch_id(depth, self.nempty)
    batch_id = torch.zeros(data.shape[0], dtype= torch.int64, device = data.device)
    ones = data.new_ones([data.shape[0], 1])
    count = scatter_add(ones, batch_id, dim=0, dim_size=batch_size)
    count = count * self.channels_per_group  # element number in each group
    inv_count = 1.0 / (count + self.eps)  # there might be 0 element sometimes

    mean = scatter_add(data, batch_id, dim=0, dim_size=batch_size) * inv_count
    mean = self._adjust_for_group(mean)
    out = data - mean.index_select(0, batch_id)

    var = scatter_add(out**2, batch_id, dim=0, dim_size=batch_size) * inv_count
    var = self._adjust_for_group(var)
    inv_std = 1.0 / (var + self.eps).sqrt()
    out = out * inv_std.index_select(0, batch_id)

    out = out * self.weights + self.bias
    return out

  def _adjust_for_group(self, tensor: torch.Tensor):
    r''' Adjust the tensor for the group.
    '''

    if self.channels_per_group > 1:
      tensor = (tensor.reshape(-1, self.group, self.channels_per_group)
                      .sum(-1, keepdim=True)
                      .repeat(1, 1, self.channels_per_group)
                      .reshape(-1, self.in_channels))
    return tensor

  def extra_repr(self) -> str:
    return ('in_channels={}, group={}, nempty={}').format(
            self.in_channels, self.group, self.nempty)  # noqa


class OctreeInstanceNorm(OctreeGroupNorm):
  r''' An instance normalization layer for the octree.
  '''

  def __init__(self, in_channels: int, nempty: bool = False):
    super().__init__(in_channels=in_channels, group=in_channels, nempty=nempty)

  def extra_repr(self) -> str:
    return ('in_channels={}, nempty={}').format(self.in_channels, self.nempty)
