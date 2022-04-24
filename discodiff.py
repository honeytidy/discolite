import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import nn
from torch.nn import functional as F
from CLIP import clip

sys.path.append('ResizeRight')
from resize_right import resize

sys.path.append('guided-diffusion')
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size,
                 Overview=4,
                 InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(input, (
            (sideY - max_size) // 2, (sideY - max_size) // 2, (sideX - max_size) // 2, (sideX - max_size) // 2))
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)

        cutouts = torch.cat(cutouts)
        cutouts = self.augs(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock([
                self.down,
                ConvBlock(cs[0], cs[1]),
                ConvBlock(cs[1], cs[1]),
                SkipBlock([
                    self.down,
                    ConvBlock(cs[1], cs[2]),
                    ConvBlock(cs[2], cs[2]),
                    SkipBlock([
                        self.down,
                        ConvBlock(cs[2], cs[3]),
                        ConvBlock(cs[3], cs[3]),
                        SkipBlock([
                            self.down,
                            ConvBlock(cs[3], cs[4]),
                            ConvBlock(cs[4], cs[4]),
                            SkipBlock([
                                self.down,
                                ConvBlock(cs[4], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[4]),
                                self.up,
                            ]),
                            ConvBlock(cs[4] * 2, cs[4]),
                            ConvBlock(cs[4], cs[3]),
                            self.up,
                        ]),
                        ConvBlock(cs[3] * 2, cs[3]),
                        ConvBlock(cs[3], cs[2]),
                        self.up,
                    ]),
                    ConvBlock(cs[2] * 2, cs[2]),
                    ConvBlock(cs[2], cs[1]),
                    self.up,
                ]),
                ConvBlock(cs[1] * 2, cs[1]),
                ConvBlock(cs[1], cs[0]),
                self.up,
            ]),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


def do_run(prompt_texts='a beautiful chinese landscape paintings',
           device='cpu', output_dir='output', count=5):
    count = int(count)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    width_height = [1280, 768]
    clip_guidance_scale = 5000
    tv_scale = 0
    range_scale = 150
    sat_scale = 0
    cutn_batches = 4
    skip_steps = 10
    side_x = (width_height[0] // 64) * 64
    side_y = (width_height[1] // 64) * 64
    cut_overview = eval('[12]*400+[4]*600')
    cut_innercut = eval('[4]*400+[12]*600')
    cut_icgray_p = eval('[0.2]*400+[0]*600')
    seed = random.randint(0, 2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    prompt_texts = [f'{prompt_texts}:1']
    normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    steps = 250
    if device == 'gpu':
        if torch.cuda.is_available():
            t = torch.cuda.get_device_properties(0).total_memory
            device = torch.device('cuda:0')
            if t < 8 * 2 ** 30:
                print(f'gpu memory ({t / 2 ** 30}G) is too low (should > 8G), using cpu')
                device = 'cpu'
            print('using device:', device)
        else:
            device = 'cpu'
            print('cuda is not available use device:', device)
    else:
        print('using device:', device)
    map_loc = 'cpu' if device == 'cpu' else None
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(torch.load(f'models/secondary_model_imagenet_2.pth', map_location=map_loc))
    secondary_model.eval().requires_grad_(False).to(device)

    clip_models = []
    clip_models.append(
        clip.load('ViT-B/32', jit=False, download_root='models', device=device)[0].eval().requires_grad_(False).to(
            device))
    clip_models.append(
        clip.load('ViT-B/16', jit=False, download_root='models', device=device)[0].eval().requires_grad_(False).to(
            device))
    clip_models.append(
        clip.load('RN50', jit=False, download_root='models', device=device)[0].eval().requires_grad_(False).to(device))

    model_config = model_and_diffusion_defaults()
    model_config.update({
        'timestep_respacing': f'ddim{steps}',
        'diffusion_steps': (1000 // steps) * steps if steps < 1000 else steps,
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'rescale_timesteps': True,
        'image_size': 512,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': True,
        'use_fp16': False,
        'use_scale_shift_norm': True,
    })
    if device == 'gpu':
        model_config['use_fp16'] = True

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(f'models/512x512_diffusion_uncond_finetune_008100.pt', map_location=map_loc))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()

    model_stats = []
    for clip_model in clip_models:
        model_stat = {"clip_model": clip_model, "target_embeds": [], "make_cutouts": None, "weights": []}
        for prompt in prompt_texts:
            txt, weight = prompt.split(':')
            weight = float(weight)
            txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
            model_stat["target_embeds"].append(txt)
            model_stat["weights"].append(weight)
        model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
        model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
        model_stat["weights"] /= model_stat["weights"].sum().abs()
        model_stats.append(model_stat)

    cur_t = None

    def cond_fn(x, t):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]
            alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
            sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
            cosine_t = alpha_sigma_to_t(alpha, sigma)
            out = secondary_model(x, cosine_t[None].repeat([n])).pred
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)

            for model_stat in model_stats:
                for i in range(cutn_batches):
                    t_int = int(t.item()) + 1  # errors on last step without +1, need to find source
                    try:
                        input_resolution = model_stat["clip_model"].visual.input_resolution
                    except:
                        input_resolution = 224

                    cuts = MakeCutoutsDango(input_resolution,
                                            Overview=cut_overview[1000 - t_int],
                                            InnerCrop=cut_innercut[1000 - t_int],
                                            IC_Size_Pow=1,
                                            IC_Grey_P=cut_icgray_p[1000 - t_int])
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                    dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                    dists = dists.view([cut_overview[1000 - t_int] + cut_innercut[1000 - t_int], n, -1])
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out)

            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale

            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if torch.isnan(x_in_grad).any() == False:
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                x_is_NaN = True
                grad = torch.zeros_like(x)
        if x_is_NaN == False:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(max=0.05) / magnitude
        return grad

    for i in range(count):
        cur_t = diffusion.num_timesteps - skip_steps - 1
        samples = diffusion.ddim_sample_loop_progressive(
            model,
            (1, 3, side_y, side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_steps,
            init_image=None,
            randomize_class=True,
            eta=0.8)
        for j, sample in enumerate(samples):
            cur_t -= 1
            for k, image in enumerate(sample['pred_xstart']):
                image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                image.save(f'{output_dir}/output{i}.png')


if __name__ == '__main__':
    do_run()
