# Set-up

## Dependencies

This codebase has been tested with the packages and versions specified in `requirements.txt` and Python 3.8.

We recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment:
```bash
conda create -n multimae python=3.8 -y
conda activate multimae
```
Then, install [PyTorch](https://pytorch.org/) 1.10.0+ and [torchvision](https://pytorch.org/vision/stable/index.html) 0.11.1+. For example:
```bash
conda install pytorch=1.10.0 torchvision=0.11.1 -c pytorch -y
```

Finally, install all other required packages:
```bash
pip install timm==0.4.12 einops==0.3.2 pandas==1.3.4
```

## Dataset Preparation

### Dataset structure

For simplicity and uniformity, all our datasets are structured in the following way:
```
/path/to/data/
├── Train/
│   ├── ADI/
│   │   ├── img1.ext1
│   │   └── img2.ext1
│   │   ...
│   │
│   └── TUM/
│       ├── img1.ext1
│       └── img2.ext1
│   
└── Test/
    ├── ADI/
    │   ├── img1.ext1
    │   └── img2.ext1
    │   ...
    │
    └── TUM/
        ├── img1.ext1
        └── img2.ext1
    
```
The folder structure and filenames should match across modalities.
After extracting H and E images following [this](#H&E Image Extraction) will create a subfolder in the following way:
```
/path/to/data/
└── Train/Test
    ├── ADI/
    │   ├── img1.ext1
    │   └── img2.ext1
    │   ...
    │
    └── mae/
        └── H
           ├── ADI_img1.ext1
           └── TUM_img2.ext1
        └── E
           ├── ADI_img2.ext1
           └── TUM_img2.ext1
```

For most experiments, we use RGB  (`rgb`), H stain image (`him`), and E stain image (`eim`) as our modalities.

RGB images are stored as either PNG or JPEG images. 
H images are single-channel PNG images. 
E images are single-channel PNG images. 

### Datasets

We use the following datasets in our experiments:
- [**NCT-CRC-HE-100K**](https://zenodo.org/record/1214456)

To download these datasets, please follow the instructions on the page. 

## H&E Image Extraction

We use the [Vahadane](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7460968) method to separate NCT-CRC-HE-100K non-background dataset. 

```bash
python -m extractHE --dataset NCK --img_size 224 --n_workers 12
```
:information_source: The MMAE pre-training strategy is flexible and can benefit from higher quality pseudo labels (H, E images) and ground truth (RGB) data.
So feel free to use other methods to decompose other histopathology stains and datasets than the ones we used!
