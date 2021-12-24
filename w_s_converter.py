# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import random
import math
import time
import click
import legacy
from typing import List, Optional

import cv2
import clip
import dnnlib
import numpy as np
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
from torch import linalg as LA
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma


def block_forward(self, x, img, ws, shapes, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter)[...,:shapes[0]], fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter)[...,:shapes[0]], fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter)[...,:shapes[1]], fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter)[...,:shapes[2]], fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--projected_s', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    projected_s: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)


    # Labels.
    label = torch.zeros([1, G.c_dim], device=device).requires_grad_()


    # Generate images.
    for i in G.parameters():
      i.requires_grad = True

    t1 = time.time()

    ws = np.load(projected_w)['w']
    ws = torch.tensor(ws, device=device)

    block_ws = []
    with torch.autograd.profiler.record_function('split_ws'):
        misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
        ws = ws.to(torch.float32)


        w_idx = 0
        for res in G.synthesis.block_resolutions:
            block = getattr(G.synthesis, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv


    styles = torch.zeros(1,26,512, device=device)
    styles_idx = 0
    temp_shapes = []
    for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
        block = getattr(G.synthesis, f'b{res}')

        if res == 4:
          temp_shape = (block.conv1.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
          styles[0,:1,:] = block.conv1.affine(cur_ws[0,:1,:])
          styles[0,1:2,:] = block.torgb.affine(cur_ws[0,1:2,:])

          block.conv1.affine = torch.nn.Identity()
          block.torgb.affine = torch.nn.Identity()
          styles_idx += 2
        else:
          temp_shape = (block.conv0.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
          styles[0,styles_idx:styles_idx+1,:temp_shape[0]] = block.conv0.affine(cur_ws[0,:1,:])
          styles[0,styles_idx+1:styles_idx+2,:temp_shape[1]] = block.conv1.affine(cur_ws[0,1:2,:])
          styles[0,styles_idx+2:styles_idx+3,:temp_shape[2]] = block.torgb.affine(cur_ws[0,2:3,:])

          block.conv0.affine = torch.nn.Identity()
          block.conv1.affine = torch.nn.Identity()
          block.torgb.affine = torch.nn.Identity()
          styles_idx += 3
        temp_shapes.append(temp_shape)


    styles = styles.detach()
    np.savez(f'{outdir}/input.npz', s=styles.cpu().numpy())

        
if __name__ == "__main__":
    generate_images() 
