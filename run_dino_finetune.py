# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import datetime
import json
import sys
import copy
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms

import baseline_models.vision_transformer as vits
from utils.general import *
from utils.dataset_histo_new import (AeSimEmbedDatasetWrapper, DistributedSamplerWrapper, DatasetFromSampler)
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

import utils.dino_utils as utils


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # ============ preparing data ... ============
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Get dataset
    args.nb_classes = 8  # opts.linprobe_n
    pathology_dataset_class = AeSimEmbedDatasetWrapper(args, transform=train_transform)
    dataset_train = pathology_dataset_class.get_dataset()

    val_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        #pth_transforms.RandomHorizontalFlip(),
        #pth_transforms.Resize(256, interpolation=3),
        #pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    vpathology_dataset_class = AeSimEmbedDatasetWrapper(args, train=False, transform=val_transform)
    dataset_test = vpathology_dataset_class.get_dataset()

    # K-fold Cross Validation model evaluation
    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y = dataset_train.get_label()
    labels = {'ADI': 0, 'DEB': 1,
              'LYM': 2, 'MUC': 3,
              'MUS': 4, 'NORM': 5,
              'STR': 6, 'TUM': 7}
    all_metrics = []
    all_avg_metrics = []
    old_bsz = copy.deepcopy(args.batch_size)
    for n_samples in args.linprobe_n:
        best_metrics = []
        args.batch_size = old_bsz * 10 if n_samples >= 100 else old_bsz
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_train, y)):
            # Balancing the sets
            tr_fold_y = y[train_ids]
            df = pd.DataFrame(dict(y=tr_fold_y, ids=train_ids))
            num_samples_per_label = n_samples // len(labels)
            df = df.groupby('y', group_keys=False).apply(
                lambda x: x.sample(min(len(x), num_samples_per_label), random_state=seed))
            train_ids = df.ids.to_numpy()

            val_fold_y = y[val_ids]
            df = pd.DataFrame(dict(y=val_fold_y, ids=val_ids))
            df = df.groupby('y', group_keys=False).apply(
                lambda x: x.sample(min(len(x), num_samples_per_label), random_state=seed))
            val_ids = df.ids.to_numpy()

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
                        print(
                            'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                            'This will slightly alter validation results as extra duplicate entries are added to achieve '
                            'equal num of samples per-process.')
                    sampler_val = DistributedSamplerWrapper(
                        valid_subsampler, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                    # TEST
                    if len(dataset_test) % num_tasks != 0:
                        print(
                            'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
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

            if utils.is_main_process():  # and args.log_wandb:
                if args.finetune != '' and not args.auto_resume:
                    # starting a finetuning experiment
                    print('starting a finetuning experiment')
                    pretrain_exp_name = os.path.basename(os.path.dirname(args.finetune))
                    chkp_name = os.path.basename(args.finetune)
                    linear_exp_name = '_'.join([
                        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        chkp_name,
                        f'samples_{n_samples}',
                        f'fold_{fold}'
                    ])
                    log_dir = os.path.join('./output_dir', 'runs', f'{pretrain_exp_name}_{linear_exp_name}')
                    args.output_dir = f'{args.finetune}_{linear_exp_name}'
                elif args.finetune != '' and args.auto_resume:
                    # resuming a finetuning experiment
                    linear_exp_name = os.path.basename(args.finetune)
                    log_dir = os.path.join('./output_dir', 'runs', f'{linear_exp_name}')
                    args.output_dir = f'{args.finetune}'
                elif args.finetune == '' and args.resume:
                    pretrain_exp_name = os.path.split(os.path.dirname(args.resume))[-1]
                    linear_exp_name = '_'.join([
                        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        str(n_samples),
                        f'fold_{fold}'
                    ])
                    log_dir = os.path.join(args.output_dir, 'runs', f'{pretrain_exp_name}_{linear_exp_name}')
                    args.output_dir = os.path.join(args.output_dir, f'{pretrain_exp_name}_{linear_exp_name}')
                else:
                    raise NotImplementedError

                if not os.path.exists(args.output_dir):
                    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                log_writer = SummaryWriter(log_dir=log_dir)
            else:
                log_writer = None

            # ============ building network ... ============
            # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
            if args.arch in vits.__dict__.keys():
                model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
                embed_dim = model.embed_dim
            else:
                print(f"Unknow architecture: {args.arch}")
                sys.exit(1)
            model.cuda()
            args.pretrained_weights = args.finetune
            # load weights to evaluate
            utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch,
                                          args.patch_size)
            print(f"Model {args.arch} built.")

            linear_classifier = LinearClassifier(embed_dim, num_labels=args.nb_classes)
            model = torch.nn.Sequential(model, linear_classifier)
            model = model.cuda()
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            print('pushed to linear')
            # set optimizer
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
                momentum=0.9,
                weight_decay=0,  # we do not apply weight decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

            # Optionally resume from a checkpoint
            to_restore = {"epoch": 0, "best_acc": 0.}
            utils.restart_from_checkpoint(
                os.path.join(args.output_dir, "checkpoint.pth.tar"),
                run_variables=to_restore,
                state_dict=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            start_epoch = to_restore["epoch"]
            best_acc = to_restore["best_acc"]
            print(to_restore)
            for epoch in range(start_epoch, args.epochs):
                data_loader_train.sampler.set_epoch(epoch)

                train_stats = train(model, optimizer, data_loader_train, epoch)
                scheduler.step()

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch}

                # if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
                val_stats = validate_network(data_loader_val, model, num_labels=args.nb_classes)
                print(
                    f"Accuracy at epoch {epoch} of the network on the {len(val_ids)} test images: {val_stats['acc1']:.1f}%")

                if utils.is_main_process():
                    log_stats = {**{f'train_{k}': v for k, v in log_stats.items()},
                                 **{f'test_{k}': v for k, v in val_stats.items()}}

                    with (Path(args.output_dir) / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    if best_acc < val_stats["acc1"]:
                        best_acc = max(best_acc, val_stats["acc1"])
                        print(f'Max accuracy so far: {best_acc:.2f}%')

                        save_dict = {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_acc": best_acc,
                        }
                        torch.save(save_dict, os.path.join(args.output_dir, "checkpoint-best.pth.tar"))
            print("Training of the supervised linear classifier on frozen features completed.\n"
                  "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
            #TEST
            if utils.is_main_process():
                # ============ building network ... ============
                # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
                if args.arch in vits.__dict__.keys():
                    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
                    embed_dim = model.embed_dim
                else:
                    print(f"Unknow architecture: {args.arch}")
                    sys.exit(1)
                model.cuda()
                print(f"Model {args.arch} built.")

                linear_classifier = LinearClassifier(embed_dim, num_labels=args.nb_classes)
                model = torch.nn.Sequential(model, linear_classifier)
                model = model.cuda()

                # resume from a checkpoint
                models_list = os.path.join(args.output_dir, "checkpoint-best.pth.tar")
                checkpoint = torch.load(models_list, map_location='cpu')
                state_dict = checkpoint['state_dict']
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                _ = model.load_state_dict(state_dict, strict=False)
                msg = f" Epoch: {checkpoint['epoch']}"
                msg += f" best_acc: {checkpoint['best_acc']}"
                print(msg, models_list)

                test_stats = validate_network(data_loader_test, model, num_labels=args.nb_classes)

                print(f"TEST: Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
                best_metrics.append(test_stats['acc1'])

        if log_writer is not None:
            average_metric = sum(best_metrics) / len(best_metrics)
            chkp_name = os.path.basename(args.finetune)
            fold_str = f'***Results for {n_samples} Samples for {chkp_name} is) {best_metrics} *** \n'
            fold_str += f'*** Validation metrics: {best_metrics} *** \n'
            fold_str += f'*** Average Overall Validation metric: {average_metric} ***'
            print(fold_str)

            all_metrics.append({n_samples: best_metrics})
            all_metrics.append({n_samples: average_metric})

            log_writer.add_text('Active_Fold_Results', str(fold_str))  # fix this as its not going up to tb
    if log_writer is not None:
        print(f"{all_metrics} \n {all_avg_metrics}")
        log_writer.add_text('Fold_Results', f"{all_metrics} \n {all_avg_metrics}")  # fix this as its not going up to tb


def train(model, optimizer, loader, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        output = model(inp)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, num_labels):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        output = model(inp)
        loss = nn.CrossEntropyLoss()(output, target)

        if num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size', default=64, type=int, help='Per-GPU batch-size')

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

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--no_dist_eval', action='store_false', dest='dist_eval',
                        help='Disabling distributed evaluation')
    parser.set_defaults(dist_eval=False)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # HISTO PARAMS
    parser.add_argument('-d', '--dataset', default='NCK', choices=['sarc', 'lusc', 'ours', 'NCK'],
                        help='dataset network must be pretrained on',
                        type=str)
    parser.add_argument('--folds', default=5, type=int, help='Number of folds to run')
    parser.add_argument('-sc', '--scale', default=20.0, choices=[1.25, 5.0, 10.0, 20.0], type=float)

    parser.add_argument('-no_t', '--no_tensorboard', action='store_true',
                        help='log to tensorboard or not, if false log to tb if True dont log')
    parser.set_defaults(no_tensorboard=False)

    parser.add_argument('-pre_HE', '--pre_extracted_h_e', action='store_true', dest='pre_extracted_h_e',
                        help='Load preextracted H and E image or Extract on the fly? default No, i.e on the fly')
    parser.set_defaults(pre_extracted_h_e=False)

    parser.add_argument('--linprobe_n', default=[96, 1000, 10000], type=int, nargs='+',
                        help='number for patches to use for training linear classifier:Default None, since pretraining')
    parser.add_argument('-est', '--early_stopping', action='store_true', default=False,
                        help='enable early stopping')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--model', default='vit_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    args = parser.parse_args()
    eval_linear(args)