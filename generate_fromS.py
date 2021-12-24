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
import PIL.Image
import matplotlib.pyplot as plt
import torch
from torch import linalg as LA
import torch.nn.functional as F
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
    """
    Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.
    """

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
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--s_input', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--text_prompt', help='Text', type=str, required=True)
@click.option('--change_power', help='Change power', type=int, required=True)
@click.option('--from_video', 'from_video', is_flag=True, help="generate from video")

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    s_input: Optional[str],
    text_prompt: str,
    change_power: int,
    from_video: bool,
):
    """
    Generate images using pretrained network pickle.

    Examples:
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
            img.save(f'{outdir}/proj{idx:02d}.png')
        return

    # Labels
    label = torch.zeros([1, G.c_dim], device=device).requires_grad_()
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images
    for i in G.parameters():
      i.requires_grad = False

    t1 = time.time()

    temp_shapes = []
    for res in G.synthesis.block_resolutions:
        block = getattr(G.synthesis, f'b{res}')
        if res == 4:
            temp_shape = (block.conv1.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
            block.conv1.affine = torch.nn.Identity()
            block.torgb.affine = torch.nn.Identity()

        else:
            temp_shape = (block.conv0.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
            block.conv0.affine = torch.nn.Identity()
            block.conv1.affine = torch.nn.Identity()
            block.torgb.affine = torch.nn.Identity()

        temp_shapes.append(temp_shape)


    if s_input is not None:
        styles = np.load(s_input)['s']
        styles_direction = np.load(f'{outdir}/direction_'+text_prompt.replace(" ", "_")+'.npz')['s']

        styles_direction = torch.tensor(styles_direction, device=device)
        styles = torch.tensor(styles, device=device)

    if from_video and not os.path.isdir(f'{outdir}_video'):
        os.makedirs(f'{outdir}_video')

    with torch.no_grad():
        if from_video:
            name_i = 1000
            for grad_change in np.arange(0, 1, 0.02)*change_power:
                imgs = []
                name_i += 1

                styles += styles_direction*grad_change
                styles_idx = 0
                x = img = None
                for k , res in enumerate(G.synthesis.block_resolutions):
                    block = getattr(G.synthesis, f'b{res}')

                    if res == 4:
                        x, img = block_forward(block, x, img, styles[:, styles_idx:styles_idx+2, :], temp_shapes[k], noise_mode=noise_mode)
                        styles_idx += 2
                    else:
                        x, img = block_forward(block, x, img, styles[:, styles_idx:styles_idx+3, :], temp_shapes[k], noise_mode=noise_mode)
                        styles_idx += 3

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
                imgs.append(img[0].to(torch.uint8).cpu().numpy())

                styles -= styles_direction*grad_change
                img_filepath = '{}_video/{}_{}_{}.jpeg'.format(outdir, text_prompt.replace(" ", "_"), change_power, name_i)
                PIL.Image.fromarray(np.concatenate(imgs, axis=1), 'RGB').save(img_filepath, quality=95)
        else:
            imgs = []
            grad_changes = [0, 0.25*change_power, 0.5*change_power, 0.75*change_power, change_power]

            for grad_change in grad_changes:
                styles += styles_direction*grad_change

                styles_idx = 0
                x = img = None
                for k , res in enumerate(G.synthesis.block_resolutions):
                    block = getattr(G.synthesis, f'b{res}')

                    if res == 4:
                        x, img = block_forward(block, x, img, styles[:, styles_idx:styles_idx+2, :], temp_shapes[k], noise_mode=noise_mode)
                        styles_idx += 2
                    else:
                        x, img = block_forward(block, x, img, styles[:, styles_idx:styles_idx+3, :], temp_shapes[k], noise_mode=noise_mode)
                        styles_idx += 3

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
                imgs.append(img[0].to(torch.uint8).cpu().numpy())

                styles -= styles_direction*grad_change

            img_filepath = f'{outdir}/'+text_prompt.replace(" ", "_")+'_'+str(change_power)+'.jpeg'
            PIL.Image.fromarray(np.concatenate(imgs, axis=1), 'RGB').save(img_filepath, quality=95)

        print("time passed:", time.time()-t1)


if __name__ == "__main__":
    generate_images()
