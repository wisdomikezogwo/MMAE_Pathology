import os

SOURCE_DATASET_NAME = os.environ['SOURCE_DATASET_NAME']

BASE_DIR = "path/to/patched/histo/datset"
PROCESSED_BASE_DIR = os.path.join(BASE_DIR, f"processed_{SOURCE_DATASET_NAME}")

WSI_BASE_DIR = os.path.join(BASE_DIR, "wsi_data")
BASE_DATASET = os.path.join(WSI_BASE_DIR, SOURCE_DATASET_NAME)
OUT_BASE = os.path.join(PROCESSED_BASE_DIR, 'pyramid')

SOURCE_DIR = os.path.join(WSI_BASE_DIR, SOURCE_DATASET_NAME)
