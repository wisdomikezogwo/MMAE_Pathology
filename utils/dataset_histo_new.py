import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, Sampler
from operator import itemgetter

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import random
from glob import glob
from general import *
from typing import Any, Optional, Sequence, Tuple, Union
import random as py_random


# from pip package catalyst
class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(torch.utils.data.DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def generate_mil_mae_csv(scale, num_images_per_patient, train_mode='pretrain', train_bool=True, task='rm',
                         subset=True, tile_size=224, multiscale=1):
    csv_path = os.path.join(OUT_BASE, f"{scale}_{tile_size}", 'scale_csv.csv')
    # args to add train_mode, train_bool, task

    if train_mode == 'pretrain':
        patients_df = pd.read_csv(csv_path)
        all_cases = patients_df['case_id'].tolist()
        patch_path = []
        for all_case in all_cases:
            if multiscale == 1:
                path_temp = os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{scale}_{tile_size}", all_case, '*.jpeg')
                glb = glob(path_temp)
                if subset:
                    if not len(glb) < num_images_per_patient:
                        glb = random.sample(glb, num_images_per_patient)
                for path in glb:
                    patch_path.append({'path': path})
            if multiscale == 0:
                path_temp = os.path.join(PROCESSED_BASE_DIR, 'single', f"{scale}_{tile_size}", all_case, '*.jpeg')
                glb = glob(path_temp)
                if subset:
                    glb = random.sample(glb, num_images_per_patient)
                patch_path.extend(glb)

        df = pd.DataFrame(patch_path)
        #if SOURCE_DATASET_NAME == 'uwsf':
        #    df.drop(df.tail(2).index, inplace=True)  # drop last n=2 rows, cause of bug i cant quite track down rn

        if multiscale == 1:
            df.to_csv(os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{scale}_{tile_size}", 'mae_patches.csv'), index=False)
        else:
            df.to_csv(os.path.join(PROCESSED_BASE_DIR, 'single', f"{scale}_{tile_size}", 'mae_patches.csv'), index=False)
    else:
        # train_mode == 'downstream'
        # for now only multiscale == 1 i.e pyramid is implemented
        patients_df = pd.read_csv(csv_path)
        if train_bool:
            patients_df = patients_df.loc[patients_df[f"{task}_train_val"] == 1]  # Train
        else:
            patients_df = patients_df.loc[patients_df[f"{task}_train_val"] == 0]  # Test
        patients_df.reset_index(drop=True, inplace=True)
        all_cases = patients_df['case_id'].tolist()

        train_csv = pd.read_csv(os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{scale}_{tile_size}", 'mae_patches.csv'))
        train_csv = train_csv.loc[:, ~train_csv.columns.str.contains('^Unnamed')]
        train_csv = train_csv.loc[:, ~train_csv.columns.str.contains('^index')]
        train_csv_list = train_csv['0'].tolist()

        patch_path = []
        for case in all_cases:
            sub_path_temp = os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{scale}_{tile_size}", case)
            pretrain_paths = [i for i in train_csv_list if sub_path_temp in i]
            path_temp = os.path.join(sub_path_temp, '*.jpeg')
            glb = glob(path_temp)
            probe_glb = list(set(glb) - set(pretrain_paths))
            if subset:
                probe_glb = random.sample(probe_glb, num_images_per_patient)
            patch_path.extend(probe_glb)

        df = pd.DataFrame(patch_path)
        if train_bool:
            df.to_csv(os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{scale}_{tile_size}", 'train_mae_patches.csv'),
                      index=False)
        else:
            df.to_csv(os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{scale}_{tile_size}", 'test_mae_patches.csv'),
                      index=False)


def generate_lim_mae_csv(dataset):
    labels = {'ADI': 0, 'DEB': 1,
              'LYM': 2, 'MUC': 3,
              'MUS': 4, 'NORM': 5,
              'STR': 6, 'TUM': 7}

    for train_or_test in ['Train', 'Test']:
        patch_path = []
        if dataset == 'NCK':
            path_temp = os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{dataset}", f'{train_or_test}', '*', '*.tif')
            glb = glob(path_temp)
            for path in glb:
                label = labels[os.path.basename(os.path.dirname(path))]
                patch_path.append({'path': path, 'label': label})

        df = pd.DataFrame(patch_path)
        df.to_csv(os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{dataset}", f'{train_or_test}', 'mae_patches.csv'), index=False)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandomGridShuffle(object):
    """
    Random shuffle grid's cells on image.
    Args:
        grid ((int, int)): size of grid for splitting image.
    Targets:
        image, mask, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, grid: Tuple[int, int] = (3, 3)):
        self.grid = grid

    def __call__(self, imgs, tiles=None):
        self.device = imgs.device
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
        to_cat = []
        for img in imgs:
            tiles = self.get_params_dependent_on_targets(img)
            if tiles is not None:
                img = self.swap_tiles_on_image(img, tiles).unsqueeze(0)
                to_cat.append(img)
        imgs = torch.cat(to_cat, dim=0)  # concat batch-wise
        return imgs

    def get_params_dependent_on_targets(self, image):
        height, width = image.shape[:2]
        n, m = self.grid

        if n <= 0 or m <= 0:
            raise ValueError("Grid's values must be positive. Current grid [%s, %s]" % (n, m))

        if n > height // 2 or m > width // 2:
            raise ValueError("Incorrect size cell of grid. Just shuffle pixels of image")

        height_split = np.linspace(0, height, n + 1, dtype=np.int)
        width_split = np.linspace(0, width, m + 1, dtype=np.int)

        height_matrix, width_matrix = np.meshgrid(height_split, width_split, indexing="ij")

        index_height_matrix = height_matrix[:-1, :-1]
        index_width_matrix = width_matrix[:-1, :-1]

        shifted_index_height_matrix = height_matrix[1:, 1:]
        shifted_index_width_matrix = width_matrix[1:, 1:]

        height_tile_sizes = shifted_index_height_matrix - index_height_matrix
        width_tile_sizes = shifted_index_width_matrix - index_width_matrix

        tiles_sizes = np.stack((height_tile_sizes, width_tile_sizes), axis=2)

        index_matrix = np.indices((n, m))
        new_index_matrix = np.stack(index_matrix, axis=2)

        for bbox_size in np.unique(tiles_sizes.reshape(-1, 2), axis=0):
            eq_mat = np.all(tiles_sizes == bbox_size, axis=2)
            new_index_matrix[eq_mat] = self.permutation(new_index_matrix[eq_mat])

        new_index_matrix = np.split(new_index_matrix, 2, axis=2)

        old_x = index_height_matrix[new_index_matrix[0], new_index_matrix[1]].reshape(-1)
        old_y = index_width_matrix[new_index_matrix[0], new_index_matrix[1]].reshape(-1)

        shift_x = height_tile_sizes.reshape(-1)
        shift_y = width_tile_sizes.reshape(-1)

        curr_x = index_height_matrix.reshape(-1)
        curr_y = index_width_matrix.reshape(-1)

        tiles = np.stack([curr_x, curr_y, old_x, old_y, shift_x, shift_y], axis=1)

        return tiles

    def swap_tiles_on_image(self, image, tiles):
        new_image = image.copy()

        for tile in tiles:
            new_image[tile[0]: tile[0] + tile[4], tile[1]: tile[1] + tile[5]] = image[tile[2]: tile[2] + tile[4],
                                                                                      tile[3]: tile[3] + tile[5]
                                                                                      ]
        return torch.tensor(new_image, device=self.device).permute(2, 0, 1)

    @staticmethod
    def permutation(
            x: Union[int, Sequence[float], np.ndarray], random_state: Optional[np.random.RandomState] = None) -> Any:
        if random_state is None:
            random_state = np.random.RandomState(py_random.randint(0, (1 << 32) - 1))
        return random_state.permutation(x)


class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img


class SubKnnDataset(Dataset):
    def __init__(self, args, dataset, indices, transform=None, H_E=False,
                 mmae=True, train=True):
        self.args = args
        self.transform = transforms.Compose([transforms.ToTensor(),
                         #transforms.Resize(size=self.args.input_size),
                            ]) if not transform else transform
        self.H_E = H_E
        self.args = args
        self.mmae = mmae
        self.train = train

        self.files_list = dataset.files_list
        self.files_list = self.files_list.iloc[indices]
        self.files_list.reset_index(drop=True, inplace=True)
        self.labels = self.files_list['label']

    def __len__(self):
        return len(self.files_list)

    def get_label(self):
        return np.array(self.files_list['label'])

    def __getitem__(self, idx):
        temp_path = self.files_list.iloc[idx, 0]  # path to c, r image
        if 'patholin2' in SERVER:
            temp_path = temp_path.replace("patho2/nobackup", "patho2nobackup", 1)

        if self.args.dataset in ['sarc', 'lusc', 'uwsf', 'mela', 'lyon']:
            label = 999
        else:
            label = self.files_list.iloc[idx, 1]

        if not self.H_E:
            img = Image.open(temp_path)
            img = self.transform(img)
            if self.mmae:
                rgb, _, _ = self.forward_images(img, train=self.train)
                tasks_img = {'rgb': rgb}
                return tasks_img, idx, label  # target
            else:
                # DINO
                return img, idx, label

    def forward_images(self, x, train=True):
        self.ae_tf, self.orient_crop_tf, self.norm_tf = _get_pipeline_transform(self.args.input_size, #224
                                                                                train=train)
        # AE
        augment_ae = T.Compose(self.ae_tf)
        #print(x.shape)
        ae_imgs = augment_ae(x) if not self.H_E else augment_ae(x[:3, ...])
        x_vhd_hi = None
        x_vhd_ei = None
        if self.H_E:
            # Compose the diff TFS
            oriented_croped = T.Compose(self.orient_crop_tf)
            norm_transform = T.Compose(self.norm_tf)

            # ------------Online Augmentation
            x_oc_i = oriented_croped(x)
            # H and E stain separated
            assert x_oc_i.shape[0] == 3 or x_oc_i.shape[0] == 5, f"C dim should be 3 or 5 but is rather {x_oc_i.shape}"
            hoc = x_oc_i[3]
            eoc = x_oc_i[4]
            x_vhd_hi = torch.unsqueeze(hoc, 0)
            x_vhd_ei = torch.unsqueeze(eoc, 0)
        return ae_imgs, x_vhd_hi, x_vhd_ei


class AeSimEmbedDataset(Dataset):
    def __init__(self, args, csv_file, n=None, seed=None, transform=None, H_E=False,
                 mmae=True, knn=False, train=True):
        self.files_list = pd.read_csv(csv_file)
        self.files_list = self.files_list.loc[:, ~self.files_list.columns.str.contains('^Unnamed')]
        self.files_list = self.files_list.loc[:, ~self.files_list.columns.str.contains('^index')]
        if n:
            self.files_list = self.files_list.sample(n=n, random_state=seed)

        self.transform = transform
        self.H_E = H_E
        self.args = args
        self.mmae = mmae
        self.knn = knn
        self.train = train

    def __len__(self):
        return len(self.files_list)

    def get_label(self):
        return np.array(self.files_list['label'])

    def __getitem__(self, idx):
        temp_path = self.files_list.iloc[idx, 0]  # path to c, r image
        if 'patholin2' in SERVER:
            temp_path = temp_path.replace("patho2/nobackup", "patho2nobackup", 1)

        if self.args.dataset in ['sarc', 'lusc', 'uwsf', 'mela', 'lyon']:
            label = 999
        else:
            label = self.files_list.iloc[idx, 1]
        if not self.knn:
            if not self.H_E:
                img = Image.open(temp_path)
                img = self.transform(img)
                if self.mmae:
                    rgb, _, _ = self.forward_images(img, train=self.train)
                    tasks_img = {'rgb': rgb}
                    return tasks_img, label  # target
                else:
                    # DINO
                    return img, label
            else:
                p = os.path.split(temp_path)
                split_on = p[0].split('/')
                save_path_H = '/'.join(split_on[:-1]) + f'/mae/H/{split_on[-1]}_' + p[1]
                save_path_E = '/'.join(split_on[:-1]) + f'/mae/E/{split_on[-1]}_' + p[1]

                img = Image.open(temp_path)
                H = Image.open(save_path_H)
                E = Image.open(save_path_E)

                new_img = np.dstack([np.array(img), np.array(H), np.array(E)])  # H, W , 5 (3 + 1 + 1)
                assert new_img.shape[-1] == 5, f'its {new_img.shape} instead'

                img = self.transform(new_img)
                #print(img.size)
                rgb, him, eim = self.forward_images(img, train=self.train)
                tasks_img = {'rgb': rgb, 'him': him, 'eim': eim}
                return tasks_img, label   # target
        else:
            if not self.H_E:
                img = Image.open(temp_path)
                img = self.transform(img)
                if self.mmae:
                    rgb, _, _ = self.forward_images(img, train=self.train)
                    tasks_img = {'rgb': rgb}
                    return tasks_img, idx, label  # target
                else:
                    # DINO
                    return img, idx, label

    def forward_images(self, x, train=True):
        self.ae_tf, self.orient_crop_tf, self.norm_tf = _get_pipeline_transform(self.args.input_size, #224
                                                                                train=train)
        # AE
        augment_ae = T.Compose(self.ae_tf + self.norm_tf)
        #print(x.shape)
        ae_imgs = augment_ae(x) if not self.H_E else augment_ae(x[:3, ...])
        x_vhd_hi = None
        x_vhd_ei = None
        if self.H_E:
            # Compose the diff TFS
            oriented_croped = T.Compose(self.orient_crop_tf)
            norm_transform = T.Compose(self.norm_tf)

            # ------------Online Augmentation
            x_oc_i = oriented_croped(x)
            # H and E stain separated
            assert x_oc_i.shape[0] == 3 or x_oc_i.shape[0] == 5, f"C dim should be 3 or 5 but is rather {x_oc_i.shape}"
            hoc = x_oc_i[3]
            eoc = x_oc_i[4]
            x_vhd_hi = torch.unsqueeze(hoc, 0)
            x_vhd_ei = torch.unsqueeze(eoc, 0)
        return ae_imgs, x_vhd_hi, x_vhd_ei


class AeSimEmbedDatasetWrapper(object):
    def __init__(self, opts, train=True, transform=None, knn=False):
        self.opts = opts
        self.scale = opts.scale
        self.input_size = opts.input_size
        self.H_E = opts.pre_extracted_h_e

        self.seed = opts.seed
        self.transform = transform
        self.knn = knn
        self.train = train

        if opts.dataset in ['sarc', 'lusc', 'uwsf', 'mela', 'lyon']:
            # i.e MIL
            self.path_to_patches = os.path.join(PROCESSED_BASE_DIR, 'pyramid',
                                                    f"{opts.scale}_{self.input_size}", 'mae_patches.csv')
        else:
            assert opts.dataset not in ['sarc', 'lusc', 'uwsf', 'mela', 'lyon', 'ImageNet']
            if train:
                self.path_to_patches = os.path.join(PROCESSED_BASE_DIR, 'pyramid',
                                                    f"NCK", 'Train', 'mae_patches.csv')
            else:
                self.path_to_patches = os.path.join(PROCESSED_BASE_DIR, 'pyramid',
                                                    f"NCK", 'Test', 'mae_patches.csv')

    def get_dataset(self):
        dataset = AeSimEmbedDataset(args=self.opts,
                                    csv_file=self.path_to_patches,
                                    seed=self.seed,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Resize(size=self.opts.input_size),
                                                                  ]) if not self.transform else self.transform,
                                    H_E=self.H_E,
                                    mmae=True if not self.transform else False,
                                    knn=self.knn,
                                    train=self.train)
        return dataset


def _get_pipeline_transform(input_size, train=True):
    # Imagenet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # simple augmentation
    if True:
        ae_transforms = [transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),  # 3 is bicubic
                         transforms.RandomHorizontalFlip(),
                         #transforms.Normalize(mean=mean, std=std)
                         ]
    else:
        ae_transforms = [transforms.Resize(256, interpolation=3),
                         transforms.CenterCrop(224),
                         #transforms.ToTensor(),
                         transforms.Normalize(mean=mean, std=std)
                         ]

    norm = [transforms.Normalize(mean=mean, std=std)]

    orient_crop_tfs = [transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),  # 3 is bicubic
                       transforms.RandomHorizontalFlip()]



    return ae_transforms, orient_crop_tfs, norm
