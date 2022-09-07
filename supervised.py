# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

import utils
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import LinearOutputAdapter
from utils import LabelSmoothingCrossEntropy, Mixup, ModelEma
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import SoftTargetCrossEntropy, accuracy, create_model
from utils.optim_factory import (LayerDecayValueAssigner, create_optimizer,)

from utils.general import *
from utils.dataset_histo_new import (AeSimEmbedDatasetWrapper,
                                     DistributedSamplerWrapper, DatasetFromSampler)
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from utils.sft_utils import EarlyStopping


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MultiMAE_olde fine-tuning and evaluation script for image classification', add_help=False)
    #############################################################################################
    parser.add_argument('-d', '--dataset', default='NCK', choices=['NCK'],
                        help='dataset network must be pretrained on',
                        type=str)
    parser.add_argument('--folds', default=5, type=int, help='Number of folds to run')
    parser.add_argument('-sc', '--scale', default=20.0, choices=[1.25, 5.0, 10.0, 20.0], type=float)

    parser.add_argument('-no_t', '--no_tensorboard', action='store_true',
                        help='log to tensorboard or not, if false log to tb if True dont log')
    parser.set_defaults(no_tensorboard=False)

    parser.add_argument('-no_pre_HE', '--no_pre_extracted_h_e', action='store_false', dest='pre_extracted_h_e',
                        help='Load preextracted H and E image or Extract on the fly? default No, i.e on the fly')
    parser.set_defaults(pre_extracted_h_e=True)

    parser.add_argument('--linprobe_n', default=None, type=int, nargs='+',
                        help='number for patches to use for training linear classifier:Default None, since pretraining')
    parser.add_argument('-est', '--early_stopping', action='store_true', default=False,
                       help='enable early stopping')
    #############################################################################################

    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Task parameters (NOT IN USE FOR NOW #WISDOM)
    parser.add_argument('--in_domains', default='rgb', type=str,
                        help='Input domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--out_domains', default='rgb', type=str,
                        help='Output domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--standardize_depth', action='store_true')
    parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
    parser.set_defaults(standardize_depth=False)

    # Model parameters
    parser.add_argument('--model', default='multivit_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='Number of global tokens to add to encoder')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='base patch size for image-like modalities')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA')

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic) (default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # Random Erase parameters
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.0)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup parameters
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Head parameters
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', default=False, action='store_true')
    parser.add_argument('--no_mean_pooling', action='store_false', dest='use_mean_pooling')
    parser.set_defaults(use_mean_pooling=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--no_dist_eval', action='store_false', dest='dist_eval',
                    help='Disabling distributed evaluation')
    parser.set_defaults(dist_eval=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=True)

    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='Log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='Project name on wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='User or team name on wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='Run name on wandb')
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    return parser.parse_args(remaining)


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    # Get dataset
    args.nb_classes = 8  #opts.linprobe_n
    pathology_dataset_class = AeSimEmbedDatasetWrapper(args, train=True)
    dataset_train = pathology_dataset_class.get_dataset()

    vpathology_dataset_class = AeSimEmbedDatasetWrapper(args, train=False)
    dataset_test = vpathology_dataset_class.get_dataset()

    # K-fold Cross Validation model evaluation
    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y = dataset_train.get_label()
    labels = {'ADI': 0, 'DEB': 1,
              'LYM': 2, 'MUC': 3,
              'MUS': 4, 'NORM': 5,
              'STR': 6, 'TUM': 7}

    if args.output_dir != './output_dir' and args.auto_resume:
        exp_name = os.path.basename(args.output_dir)
    elif args.output_dir == './output_dir' and args.resume:
        exp_name = os.path.split(os.path.dirname(args.resume))[-1]
    else:
        exp_name = '_'.join([
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
            f"supervised",
            str(args.input_size),
            "fold_0"
        ])

    best_metrics = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_train, y)):
        exp_name = f"{exp_name[:-1]}{fold}"

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        args.distributed = False
        if 'WORLD_SIZE' in os.environ:
            # this would be in there thanks to torch_run or in our case rn torch.distributed.launch
            args.distributed = int(os.environ['WORLD_SIZE']) > 1

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            sampler_train = DistributedSamplerWrapper(
                train_subsampler, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                # VALIDATION
                if len(DatasetFromSampler(valid_subsampler)) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                          'This will slightly alter validation results as extra duplicate entries are added to achieve '
                          'equal num of samples per-process.')
                sampler_val = DistributedSamplerWrapper(
                    valid_subsampler, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                # TEST
                if len(dataset_test) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                          'This will slightly alter validation results as extra duplicate entries are added to achieve '
                          'equal num of samples per-process.')
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = valid_subsampler
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            global_rank = 0  # in the case of cpu or cuda but not distributed (me)
            sampler_train = train_subsampler
            sampler_val = valid_subsampler
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        if dataset_test is not None:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
        else:
            data_loader_test = None

        if global_rank == 0:  # and args.log_wandb:
            if args.output_dir != './output_dir' and args.auto_resume:
                # Not in use
                log_dir = os.path.join('./output_dir', 'runs', exp_name)
            elif args.output_dir == './output_dir' and args.resume:
                #Not in use
                log_dir = os.path.join(args.output_dir, 'runs', exp_name)
                args.output_dir = os.path.join(args.output_dir, exp_name)
            else:
                log_dir = os.path.join('./output_dir', 'runs', exp_name)
                args.output_dir = os.path.join('./output_dir', exp_name)

            if not os.path.exists(args.output_dir):
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            log_writer = SummaryWriter(log_dir=log_dir)
        else:
            log_writer = None

        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)

        input_adapters = {
            'rgb': PatchedInputAdapter(
                num_channels=3, stride_level=1,
                patch_size_full=args.patch_size,
                image_size=args.input_size
            )
        }
        output_adapters = {
            'cls': LinearOutputAdapter(
                num_classes=args.nb_classes,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale
            )
        }
        model = create_model(
            args.model,
            input_adapters=input_adapters,
            output_adapters=output_adapters,
            num_global_tokens=args.num_global_tokens,
            drop_rate=args.drop,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_path,
        )

        model.to(device)

        model_ema = None
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')
            print("Using EMA with decay = %.8f" % args.model_ema_decay)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print(f"Number of params: {n_parameters / 1e6} M")

        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        num_training_steps_per_epoch = len(train_ids) // total_batch_size
        args.lr = args.blr * total_batch_size / 256
        print("LR = %.8f" % args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % args.update_freq)
        print("Number of training examples = %d" % len(train_ids))
        print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

        num_layers = model_without_ddp.get_num_layers()
        if args.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(
                list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = model.no_weight_decay()
        print("Skip weight decay list: ", skip_weight_decay_list)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

        print("Use step level LR scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        print("criterion = %s" % str(criterion))

        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

        if args.eval:
            test_stats = evaluate(data_loader_test, model, device)
            print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
            exit(0)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0

        if args.early_stopping:
            early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
        else:
            early_stopping = None

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            )

            val_stats = evaluate(data_loader_val, model, device, header='Val:')
            print(f"Accuracy of the network on the {len(val_ids)} test images: {val_stats['acc1']:.1f}%")
            if max_accuracy < val_stats["acc1"]:
                max_accuracy = val_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            '''
            if log_writer is not None:
                epoch_1000x = (int(epoch) * 1000)
                for name, value in log_stats.items():
                    log_writer.add_scalar(f'{name}', value, epoch_1000x)
            '''
            if args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if early_stopping:
                early_stopping(epoch, val_stats['loss'])
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        test_input_adapters = {
            'rgb': PatchedInputAdapter(
                num_channels=3, stride_level=1,
                patch_size_full=args.patch_size,
                image_size=args.input_size
            )
        }
        test_output_adapters = {
            'cls': LinearOutputAdapter(
                num_classes=args.nb_classes,
                use_mean_pooling=args.use_mean_pooling,
                init_scale=args.init_scale
            )
        }
        if global_rank == 0:
            models_list = os.path.join(args.output_dir, 'checkpoint-best.pth')
            # Test Model
            model = create_model(
                args.model,
                input_adapters=test_input_adapters,
                output_adapters=test_output_adapters,
                num_global_tokens=args.num_global_tokens,
                drop_rate=args.drop,
                attn_drop_rate=args.attn_drop_rate,
                drop_path_rate=args.drop_path,
            )
            checkpoint = torch.load(models_list, map_location='cpu')
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            print(msg)

            model.to(device)

            test_stats = evaluate(data_loader_test, model, device)
            print(f"TEST: Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
            best_metrics.append(test_stats['acc1'])
            if log_writer is not None:
                log_writer.add_text('Each_Fold_Results', str(test_stats['acc1']))  # fix this as its not going up to tb

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    if log_writer is not None:
        average_metric = sum(best_metrics) / len(best_metrics)
        chkp_name = os.path.basename(exp_name)
        fold_str = f'***Results for Samples for {chkp_name} is) {best_metrics} *** \n'
        fold_str += f'*** Validation metrics: {best_metrics} *** \n'
        fold_str += f'*** Average Overall Validation metric: {average_metric} ***'
        print(fold_str)

        log_writer.add_text('Fold_Results', str(fold_str))  # fix this as its not going up to tb


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    outputs = outputs['cls']
    loss = criterion(outputs, target)
    return loss, outputs


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples['rgb'].to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            loss, output = train_class_batch(
                model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_this = {
                        'loss': loss_value,
                        'lr': max_lr,
                        'weight_decay': weight_decay_value,
                        'grad_norm': grad_norm,
                        }
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
            for name, value in log_this.items():
                log_writer.add_scalar(f'{name}', value, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:'):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 50, header):
        images = batch[0]['rgb']
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            output = output['cls']
            loss = criterion(output, target)

        acc1, acc3 = accuracy(output, target, topk=(1, 3))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
    # gather the stats from all processes
    if args.dist_eval:
        metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top3=metric_logger.acc3, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
