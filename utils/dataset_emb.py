import numpy as np

import pandas as pd
from PIL import Image

from einops import rearrange
from general import *


np.random.seed(0)


def channel_last(tensor):
    return rearrange(tensor, 'c a b -> a b c')


def channel_first(tensor):
    return rearrange(tensor, 'a b c -> c a b')


def set_dims(mae_model):
    name = mae_model.split('_')[-1]
    if name == 'small':
        dim_tokens = 384
    elif name == 'base':
        dim_tokens = 768
    elif name == 'large':
        dim_tokens = 1024
    else:
        raise NotImplementedError
    return dim_tokens


class EmbedDataset():
    # just to get the dataset i.e images and tab data and pass
    # them through a feature extractor like cnn, Linear or TabNet
    # has to be able to pass in scale
    def __init__(self, base, case_id, scale, patch_size, transform=None, mil=True, vhd='', heirarchy=False):
        # csv has to have patches in order so downstream pos_embeding is easy
        if mil:
            self.vhd = vhd
            opt = f'{base}/{scale}_{patch_size}/scale_csv.csv'
            self.patients_df = pd.read_csv(opt)
            self.patients_df = self.patients_df.set_index('case_id')

            self.case_tile_df = pd.read_csv(self.patients_df['tile_path'][case_id], index_col=0)
            # remove rows with none as slice index
            #self.case_tile_df.dropna(axis=0, subset=['slice_index'], inplace=True)
            # might be redundant soon
            self.case_tile_df = self.case_tile_df.loc[:, ~self.case_tile_df.columns.str.contains('^Unnamed')]
            self.case_tile_df = self.case_tile_df.loc[:, ~self.case_tile_df.columns.str.contains('^index')]
            if heirarchy:
                self.case_tile_df.sort_values(['row', 'col'], ascending=[True, True], inplace=True)
            else:
                self.case_tile_df.sort_values(['col', 'row'], ascending=[True, True], inplace=True)
            self.case_tile_df.reset_index(drop=True, inplace=True)
        else:
            self.path_to_patches = os.path.join(base, f"NCK", case_id, 'mae_patches.csv')
            self.case_tile_df = self.case_tile_df.loc[:, ~self.case_tile_df.columns.str.contains('^Unnamed')]
            self.case_tile_df = self.case_tile_df.loc[:, ~self.case_tile_df.columns.str.contains('^index')]

        self.transform = transform
        self.mil = mil

    def __getitem__(self, idx):
        # num_patches = len(self.case_tile_df)
        if self.mil:
            temp_path = self.case_tile_df.loc[idx, 'path']  # path to c, r image
            if self.vhd:
                split_j = temp_path.split('/')
                temp_path = '/'.join([f'{subdir}_norm' if subdir == 'pyramid' else subdir for subdir in split_j])

            if 'patholin2' in SERVER:
                temp_path = temp_path.replace("patho2/nobackup", "patho2nobackup", 1)

            col_row_mame = f"{self.case_tile_df.loc[idx, 'col']}_{self.case_tile_df.loc[idx, 'row']}"
            img = Image.open(temp_path)

            if self.transform:
                img = self.transform(img)
            return img, col_row_mame
        else:
            temp_path = self.case_tile_df.iloc[idx, 0]  # path to c, r image
            label = self.case_tile_df.iloc[idx, 1]
            if 'patholin2' in SERVER:
                temp_path = temp_path.replace("patho2/nobackup", "patho2nobackup", 1)

            img = Image.open(temp_path)

            if self.transform:
                img = self.transform(img)
            return img, label

    def __len__(self):
        return len(self.case_tile_df)
