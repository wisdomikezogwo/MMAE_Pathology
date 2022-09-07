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
import os
import sys
import argparse
import datetime
from pathlib import Path
import pandas as pd

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms

import baseline_models.vision_transformer as vits
from utils.general import *
from utils.dataset_histo_new import (AeSimEmbedDatasetWrapper,
                                     DatasetFromSampler,
                                     SubKnnDataset)
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import OutputAdapter
from utils import create_model
from utils.dataset_emb import set_dims

import utils.dino_utils as utils


def get_model(args):
    if "multi" in args.model:
        return get_mmae_model(args)
    else:
        return get_dino_model(args)


def get_dino_model(args):
    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    return model, args


def get_mmae_model(args):
    input_adapters = {
        'rgb': PatchedInputAdapter(
            num_channels=3, stride_level=1,
            patch_size_full=args.patch_size,
            image_size=args.input_size
        )
    }
    output_adapters = {
        'cls': OutputAdapter(use_mean_pooling=args.global_pool)
    }
    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=1,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    )

    checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    model = model.cuda()
    #model.eval()
    #for _, p in model.named_parameters():
    #    p.requires_grad = False
    args.emb_dim = set_dims(args.model)
    return model, args


def extract_test_feature_pipeline(args):
    # ============ building network ... ============
    model, args = get_model(args)
    model.eval()

    # ============ preparing data ... ============
    if "multi" in args.model:
        transform = None
        args.dino = False
    else:
        transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224),
            #pth_transforms.Resize(256, interpolation=3),
            #pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        args.dino = True

    args.nb_classes = 8  # opts.linprobe_n
    vpathology_dataset_class = AeSimEmbedDatasetWrapper(args, train=False, transform=transform, knn=True)
    dataset_val = vpathology_dataset_class.get_dataset()
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print("Extracting features for val set...")
    test_features, te_labels = extract_features(model, data_loader_val, args.use_cuda, args.dino)
    if utils.get_rank() == 0:
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    test_labels = torch.tensor(te_labels).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return test_features, test_labels


def extract_feature_pipeline(args, n_samples):
    # ============ building network ... ============
    model, args = get_model(args)
    model.eval()

    # ============ preparing data ... ============
    if "multi" in args.model:
        transform = None
        args.dino = False
    else:
        transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224),
            #pth_transforms.Resize(256, interpolation=3),
            #pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        args.dino = True

    args.nb_classes = 8  # opts.linprobe_n
    pathology_dataset_class = AeSimEmbedDatasetWrapper(args, transform=transform, knn=True)
    dataset_train = pathology_dataset_class.get_dataset()

    # K-fold Cross Validation model evaluation
    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y = dataset_train.get_label()
    labels = {'ADI': 0, 'DEB': 1,
              'LYM': 2, 'MUC': 3,
              'MUS': 4, 'NORM': 5,
              'STR': 6, 'TUM': 7}

    train_features_list = []
    train_labels_list = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_train, y)):
        print(f"Fold: {fold} Feature Extraction")
        # Balancing the sets
        tr_fold_y = y[train_ids]
        df = pd.DataFrame(dict(y=tr_fold_y, ids=train_ids))
        num_samples_per_label = n_samples // len(labels)
        df = df.groupby('y', group_keys=False).apply(
            lambda x: x.sample(min(len(x), num_samples_per_label), random_state=seed))
        train_ids = df.ids.to_numpy()

        sub_dataset_train = SubKnnDataset(args, dataset=dataset_train, indices=train_ids,
                                        transform=transform, H_E=False, mmae=not args.dino)
        sampler = torch.utils.data.DistributedSampler(sub_dataset_train, shuffle=False)

        print("Sampler_train = %s" % str(sampler))

        data_loader_train = torch.utils.data.DataLoader(
            sub_dataset_train,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        print(f"Data loaded with {len(sub_dataset_train)} train imgs.")

        # ============ extract features ... ============
        print("Extracting features for train set...")
        train_features, tr_labels = extract_features(model, data_loader_train, args.use_cuda, args.dino)

        if utils.get_rank() == 0:
            train_features = nn.functional.normalize(train_features, dim=1, p=2)

        train_labels = torch.tensor(tr_labels).long()
        # save features and labels
        if args.dump_features and dist.get_rank() == 0:
            torch.save(train_features.cpu(), os.path.join(args.dump_features, f"trainfeat_{n_samples}_{fold}.pth"))
            torch.save(train_labels.cpu(), os.path.join(args.dump_features, f"trainlabels_{n_samples}_{fold}.pth"))

            train_features_list.append(train_features)
            train_labels_list.append(train_labels)

    return train_features_list, train_labels_list


def extract_all_feature_pipeline(args, n_samples):
    # ============ building network ... ============
    model, args = get_model(args)
    model.eval()

    # ============ preparing data ... ============
    if "multi" in args.model:
        transform = None
        args.dino = False
    else:
        transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224),
            #pth_transforms.Resize(256, interpolation=3),
            #pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        args.dino = True

    args.nb_classes = 8  # opts.linprobe_n
    pathology_dataset_class = AeSimEmbedDatasetWrapper(args, transform=transform, knn=True)
    dataset_train = pathology_dataset_class.get_dataset()

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)

    print("Sampler_train = %s" % str(sampler))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Data loaded with {len(dataset_train)} train imgs.")

    # ============ extract features ... ============
    print("Extracting features for all train set...")
    train_features_list = []
    train_labels_list = []
    train_features, tr_labels = extract_features(model, data_loader_train, args.use_cuda, args.dino)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
    train_labels = torch.tensor(tr_labels).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, f"trainfeat_{n_samples}_{0}.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, f"trainlabels_{n_samples}_{0}.pth"))

        train_features_list.append(train_features)
        train_labels_list.append(train_labels)
    return train_features_list, train_labels_list


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, is_dino=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    labels = []
    #print(next(iter(data_loader)))
    for samples, index, label in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True) if is_dino else samples['rgb'].cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        #print(label, index)
        labels.extend(label.cpu().tolist())

        if multiscale and is_dino:
            feats = utils.multi_scale(samples, model)
        elif not multiscale and is_dino:
            #DINO
            feats = model(samples).clone()
        else:
            #MMAE
            feats = model(samples)
            feats = feats['cls'].clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(DatasetFromSampler(data_loader.sampler)), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features, labels


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=8):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # Dataset parameters
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--dump_or_load', default='load', choices=['load', 'dump'],
        help='str to save or load computed features, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)
    # HISTO PARAMS
    parser.add_argument('-d', '--dataset', default='NCK', choices=['sarc', 'uwsf', 'NCK'],
                        help='dataset network must be pretrained on',type=str)
    parser.add_argument('--folds', default=5, type=int, help='Number of folds to run')
    parser.add_argument('-sc', '--scale', default=20.0, choices=[1.25, 5.0, 10.0, 20.0], type=float)
    parser.add_argument('--linprobe_n', default=[96, 1000], type=int, nargs='+',
                        help='number for patches to use for training linear classifier:Default None, since pretraining')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--model', default='vit_small', type=str, choices=['vit_small', 'multivit_mid'], metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('-pre_HE', '--pre_extracted_h_e', action='store_true',
                        help='Load preextracted H and E image or Extract on the fly? default No, i.e on the fly')
    parser.set_defaults(pre_extracted_h_e=False)

    parser.add_argument('-gap', '--global_pool', action='store_true', help='global_pool')
    parser.set_defaults(global_pool=False)
    args = parser.parse_args()

    args.arch = args.model
    c_name = os.path.basename(os.path.dirname(args.pretrained_weights))
    if args.dump_or_load == 'laod':
        args.load_features = os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"NCK", c_name)
        args.dump_features = ''
    else:
        args.dump_features = os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"NCK", c_name)
        args.load_features = ''
        Path(args.dump_features).mkdir(parents=True, exist_ok=True)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    seed = args.seed + utils.get_rank()
    all_metrics = []
    for n_samples in args.linprobe_n:
        best_metrics = {k:[] for k in args.nb_knn}
        do_all = True if n_samples > 3000 else False
        args.folds = 1 if do_all else args.folds
        n_samples = 'all' if do_all else n_samples

        if args.load_features:
            train_features = []
            train_labels = []
            for fold in range(args.folds):
                train_features.append(torch.load(os.path.join(args.load_features, f"trainfeat_{n_samples}_{fold}.pth")))
                train_labels.append(torch.load(os.path.join(args.load_features, f"trainlabels_{n_samples}_{fold}.pth")))
            test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
            test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
        else:
            # need to extract features !
            if not do_all:
                train_features, train_labels = extract_feature_pipeline(args, n_samples)
            else:
                train_features, train_labels = extract_all_feature_pipeline(args, n_samples)
            test_features, test_labels = extract_test_feature_pipeline(args)

        if utils.get_rank() == 0:
            if args.use_cuda:
                train_features = [ft.cuda() for ft in train_features]
                train_labels = [ft.cuda() for ft in train_labels]
                test_features = test_features.cuda()
                test_labels = test_labels.cuda()

            print("Features are ready!\nStart the k-NN classification.")
            for fold in range(args.folds):
                if utils.get_rank() == 0:  # and args.log_wandb:
                    if args.pretrained_weights != '':
                        print('starting a knn experiment')
                        pretrain_exp_name = os.path.basename(os.path.dirname(args.pretrained_weights))
                        chkp_name = os.path.basename(args.pretrained_weights)
                        linear_exp_name = '_'.join([
                            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                            chkp_name, 'knn', f'samples_{n_samples}', f'fold_{fold}'
                        ])
                        log_dir = os.path.join('./output_dir', 'runs', f'{pretrain_exp_name}_{linear_exp_name}')
                        args.output_dir = f'{args.pretrained_weights}_{linear_exp_name}'
                    else:
                        raise NotImplementedError
                    if not os.path.exists(args.output_dir):
                        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                    log_writer = SummaryWriter(log_dir=log_dir)
                else:
                    log_writer = None

                for k in args.nb_knn:
                    top1, top5 = knn_classifier(train_features[fold], train_labels[fold],
                        test_features, test_labels, k, args.temperature)
                    print(f"For Sample size {n_samples} ::: {k}-NN classifier result: Top1: {top1}, Top5: {top5}")
                    log_writer.add_text('KNN', f"For Sample size {n_samples} ::: {k}-NN classifier result: Top1: {top1},"
                                               f" Top5: {top5}")
                    best_metrics[k].append(top1)

            if log_writer is not None:

                average_metric = {k: sum(best_metrics[k]) / len(best_metrics[k]) for k in args.nb_knn}
                chkp_name = os.path.basename(args.pretrained_weights)
                for k in args.nb_knn:
                    fold_str = f'***Results for {n_samples} Samples for KNN with {k} NN is {best_metrics[k]} *** \n'
                    fold_str += f'*** Validation metrics: {best_metrics[k]} *** \n'
                    fold_str += f'*** Average Overall Validation metric: {average_metric[k]} ***'
                    print(fold_str)

                all_metrics.append({n_samples: best_metrics})
                all_metrics.append({n_samples: average_metric})

                log_writer.add_text('Active_Fold_Results', str(fold_str))  # fix this as its not going up to tb

    all_avg_metrics = []
    if log_writer is not None and utils.get_rank() == 0:
        print(f"{all_metrics} \n {all_avg_metrics}")
        log_writer.add_text('Fold_Results', f"{all_metrics} \n {all_avg_metrics}")  # fix this as its not going up to tb

    dist.barrier()


