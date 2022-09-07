from typing import Iterable, List

import torch
import numpy as np
from einops import rearrange

import torchvision
import torchvision.transforms.functional as TF

IMAGENET_DEFAULT_MEAN = mean = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = std = np.array([0.229, 0.224, 0.225])


def visualize_(model: torch.nn.Module, data_loader: Iterable, device: torch.device,
              epoch: int, log_writer=None, num_encoded_tokens: int = 196, in_domains: List[str] = [],
              out_domains: List[str] = [], alphas: List[float] = [1.0], sample_tasks_uniformly: bool = False,
              standardize_depth: bool = True, fp32_output_adapters: List[str] = [], img_size=224, patch_size=16):

    model.eval()
    x, _ = next(iter(data_loader))  # x has 5 or 3 channels

    tasks_dict = {
        task: tensor.to(device, non_blocking=True)
        for task, tensor in x.items()
    }

    # Truncated depth standardization
    standardize_depth = False
    if standardize_depth and 'him' in tasks_dict:
        # Flatten depth and remove bottom and top 10% of values
        trunc_depth = torch.sort(rearrange(tasks_dict['him'], 'b c h w -> b (c h w)'), dim=1)[0]
        trunc_depth = trunc_depth[:, int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
        tasks_dict['him'] = (tasks_dict['him'] - trunc_depth.mean(dim=1)[:, None, None, None]) / torch.sqrt(
            trunc_depth.var(dim=1)[:, None, None, None] + 1e-6)

        trunc_depth = torch.sort(rearrange(tasks_dict['eim'], 'b c h w -> b (c h w)'), dim=1)[0]
        trunc_depth = trunc_depth[:, int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
        tasks_dict['eim'] = (tasks_dict['eim'] - trunc_depth.mean(dim=1)[:, None, None, None]) / torch.sqrt(
            trunc_depth.var(dim=1)[:, None, None, None] + 1e-6)

    input_dict = {
        task: tensor
        for task, tensor in tasks_dict.items()
        if task in in_domains
    }

    with torch.no_grad():
        preds, masks = model(
            input_dict,
            num_encoded_tokens=num_encoded_tokens,
            alphas=alphas,
            sample_tasks_uniformly=sample_tasks_uniformly,
            fp32_output_adapters=fp32_output_adapters
        )
        preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
        masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

        img_grid, him_grid, eim_grid = plot_predictions(input_dict, preds, masks, out_domains=out_domains,
                                                        image_size=img_size, patch_size=patch_size)

        log_writer.add_image('RGB', img_grid, epoch)  # (same order as in mae paper)
        if 'him' in out_domains:
            log_writer.add_image('H_IMG', him_grid, epoch)  # (same order as in mae paper)
        if 'eim' in out_domains:
            log_writer.add_image('E_IMG', eim_grid, epoch)  # (same order as in mae paper)
    return


def get_masked_image(img, mask, image_size=224, patch_size=16, mask_value=0.0):
    img_token = rearrange(
        img.detach().cpu(),
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)',
        ph=patch_size, pw=patch_size, nh=image_size // patch_size, nw=image_size // patch_size
    )
    img_token[mask.detach().cpu() != 0] = mask_value
    img = rearrange(
        img_token,
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)',
        ph=patch_size, pw=patch_size, nh=image_size // patch_size, nw=image_size // patch_size
    )
    return img


def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(),
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )


def get_pred_with_input(gt, pred, mask, image_size=224, patch_size=16):
    gt_token = rearrange(
        gt.detach().cpu(),
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)',
        ph=patch_size, pw=patch_size, nh=image_size // patch_size, nw=image_size // patch_size
    )
    pred_token = rearrange(
        pred.detach().cpu(),
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)',
        ph=patch_size, pw=patch_size, nh=image_size // patch_size, nw=image_size // patch_size
    )
    pred_token[mask.detach().cpu() == 0] = gt_token[mask.detach().cpu() == 0]
    img = rearrange(
        pred_token,
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)',
        ph=patch_size, pw=patch_size, nh=image_size // patch_size, nw=image_size // patch_size
    )
    return img


def plot_predictions(input_dict, preds, masks, out_domains, image_size=224, patch_size=16):

    masked_rgb = get_masked_image(denormalize(input_dict['rgb']),
                                  masks['rgb'],
                                  image_size=image_size,
                                  mask_value=1.0,
                                  patch_size=patch_size)[0].permute(1, 2, 0).detach().cpu()

    pred_rgb2 = get_pred_with_input(denormalize(input_dict['rgb']),
                                    denormalize(preds['rgb']).clamp(0, 1),
                                    masks['rgb'],
                                    image_size=image_size,
                                    patch_size=patch_size)[0].permute(1, 2, 0).detach().cpu()
    # bchw, and b=1
    rgb_imgs = [masked_rgb.permute(2, 0, 1), pred_rgb2.permute(2, 0, 1),
                denormalize(input_dict['rgb'])[0].detach().cpu()]
    rgb_grid = torchvision.utils.make_grid(rgb_imgs)

    him_grid, eim_grid = None, None

    if 'him' in out_domains:
        masked_him = get_masked_image(input_dict['him'],
                                    masks['him'],
                                    image_size=image_size,
                                    mask_value=np.nan,
                                    patch_size=patch_size)[0, 0].detach().cpu()
        pred_him2 = get_pred_with_input(input_dict['him'],
                                        preds['him'],
                                        masks['him'],
                                        image_size=image_size,
                                        patch_size=patch_size)[0, 0].detach().cpu()

        # channel here is 1, so its 224x224
        masked_him = torch.vstack(
            (torch.unsqueeze(masked_him, 0), torch.unsqueeze(masked_him, 0), torch.unsqueeze(masked_him, 0)))
        pred_him2 = torch.vstack(
            (torch.unsqueeze(pred_him2, 0), torch.unsqueeze(pred_him2, 0), torch.unsqueeze(pred_him2, 0)))
        orig_him = input_dict['him'][0, 0].detach().cpu()
        orig_him = torch.vstack(
            (torch.unsqueeze(orig_him, 0), torch.unsqueeze(orig_him, 0), torch.unsqueeze(orig_him, 0)))
        him_imgs = [masked_him, pred_him2, orig_him]
        him_grid = torchvision.utils.make_grid(him_imgs)  # [3, 228, 680]; padding of 2 is why its not [3, 224, 672]

    if 'eim' in out_domains:
        masked_eim = get_masked_image(input_dict['eim'],
                           masks['eim'],
                           image_size=image_size,
                           mask_value=np.nan,
                           patch_size=patch_size)[0, 0].detach().cpu()

        pred_eim2 = get_pred_with_input(input_dict['eim'],
                                      preds['eim'],
                                      masks['eim'],
                                      image_size=image_size,
                                      patch_size=patch_size)[0, 0].detach().cpu()

        masked_eim = torch.vstack((torch.unsqueeze(masked_eim, 0), torch.unsqueeze(masked_eim, 0), torch.unsqueeze(masked_eim, 0)))
        pred_eim2 = torch.vstack((torch.unsqueeze(pred_eim2, 0), torch.unsqueeze(pred_eim2, 0), torch.unsqueeze(pred_eim2, 0)))
        orig_eim = input_dict['eim'][0, 0].detach().cpu()
        orig_eim = torch.vstack((torch.unsqueeze(orig_eim, 0), torch.unsqueeze(orig_eim, 0), torch.unsqueeze(orig_eim, 0)))

        eim_imgs = [masked_eim, pred_eim2, orig_eim]
        eim_grid = torchvision.utils.make_grid(eim_imgs)


    return rgb_grid, him_grid, eim_grid

