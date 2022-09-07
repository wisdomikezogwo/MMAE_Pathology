import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import time

from utils.general import *
from utils.dataset_histo_new import generate_mil_mae_csv, generate_lim_mae_csv
from utils.vahadane import read_image, Vahadane

import multiprocessing.pool
from contextlib import closing
from itertools import product
import argparse


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.


class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


def main(df, perturb):
    pvhd = Vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=0, getH_mode=0, ITER=100, perturb=perturb)

    for ind, row in df.iterrows():
        try:
            path = row['0']
            p = os.path.split(path)
            split_on = p[0].split('/')
            save_path_H = '/'.join(split_on[:-1]) + f'/mae/H/{split_on[-1]}_' + p[1]
            save_path_E = '/'.join(split_on[:-1]) + f'/mae/E/{split_on[-1]}_' + p[1]
            if not os.path.exists(save_path_H) or not os.path.exists(save_path_E):
                print(f"doing {ind}")
                x = read_image(path)
                W, H = pvhd.stain_separate(x)
                img_H, img_E = pvhd.H_E(x, H)
                img_H = img_H.squeeze()  # H, W
                img_E = img_E.squeeze()

                img_H = Image.fromarray(img_H)
                img_H.save(save_path_H)

                img_E = Image.fromarray(img_E)
                img_E.save(save_path_E)
            else:
                print(f"Done {ind} before")
        except Exception as e:
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('H, E  extraction', add_help=False)
    parser.add_argument('-d', '--dataset', default='sarc', choices=['ImageNet', 'sarc', 'brca', 'NCK'],
                        help='dataset network must be pretrained on', type=str)
    parser.add_argument('-sc', '--scale', default=20.0, choices=[1.25, 5.0, 10.0, 20.0], type=float)
    parser.add_argument('--perturb', action='store_true', help='perturb W? default: No')
    parser.set_defaults(perturb=False)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--n_workers', default=24, type=int)
    parser.add_argument('-mp', '--multiprocess', action='store_true', help='multiprocess this code? default: Yes')
    parser.set_defaults(multiprocess=False)
    parser.add_argument('-nmil', '--not_mil', action='store_true', help='is this a MIL dataset or not. default: Not')
    parser.set_defaults(not_mil=False)

    parser.add_argument('--chunk', default=0, choices=[0, 1, 2, 3, 4], type=int, help='what chunk of the bug df to process')
    args = parser.parse_args()

    assert args.dataset == SOURCE_DATASET_NAME

    if not args.not_mil:
        # i.e MIL
        train_images_path = os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{args.scale}_{args.img_size}", 'mae_patches.csv')

        if not os.path.exists(train_images_path):
            generate_mil_mae_csv(args.scale, 2000, train_mode='pretrain',
                                 subset=True, tile_size=args.img_size, multiscale=1)

        mae_img_path = os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{args.scale}_{args.img_size}", 'mae')

        if mae_img_path:
            Path(os.path.join(mae_img_path, 'H')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(mae_img_path, 'E')).mkdir(parents=True, exist_ok=True)

        df_ = pd.read_csv(train_images_path)
        dfs = np.array_split(df_, 5)  # split the dataframe into 161 separate tables
        df = dfs[args.chunk]

        length = len(df)
        print(f"Length of df is {length} and mp is {args.multiprocess}")
        time.sleep(8)

        if args.multiprocess:
            length = len(df)
            div = length // args.n_workers
            splits = [df.iloc[i * div: (i + 1) * div] for i in range(args.n_workers)]

            with closing(NestablePool(processes=args.n_workers)) as pool:
                pool.starmap(main, product(splits, [args.perturb]))

            print('Extraction done with Multiprocess.')
        else:
            main(df, args.perturb)
            print('Extraction done.')
    else:
        for train_or_test in ['Train', 'Test']:
            if args.dataset == 'NCK':
                images_path = os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{args.dataset}", f'{train_or_test}', 'mae_patches.csv')
                mae_img_path = os.path.join(PROCESSED_BASE_DIR, 'pyramid', f"{args.dataset}", f'{train_or_test}', 'mae')

                if mae_img_path:
                    Path(os.path.join(mae_img_path, 'H')).mkdir(parents=True, exist_ok=True)
                    Path(os.path.join(mae_img_path, 'E')).mkdir(parents=True, exist_ok=True)

                if not os.path.exists(images_path):
                    generate_lim_mae_csv(args.dataset)

                df = pd.read_csv(images_path)
                length = len(df)
                print(f"Length of df is {length} and mp is {args.multiprocess}")
                time.sleep(3)

                if args.multiprocess:
                    length = len(df)
                    div = length // args.n_workers
                    splits = [df.iloc[i * div: (i + 1) * div] for i in range(args.n_workers)]

                    with closing(NestablePool(processes=args.n_workers)) as pool:
                        pool.starmap(main, product(splits, [args.perturb]))

                    print('Extraction done with Multiprocess.')
                else:
                    main(df, args.perturb)
                    print('Extraction done.')
