# Pre-training

We provide MMAE pre-training scripts on (multi-modal) NCT-CRC-HE-100K.  

All our models are pre-trained on a single node with **4 A4000 GPUs**. 

To pre-train MultiMAE on 4 GPUs using default settings, run:
```bash
# MAE RGB Only 
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --master_port=6060 run_pretraining_multimae.py \
--dist_url env://mmae/ -no_pre_HE --in_domains rgb --out_domains rgb --model pretrain_multimae_small \
--num_encoded_tokens 98 --batch_size 78 


# MMAE: RGB, H, and E and alphas for H and E importance and one decoder rgb and one modality per patch
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --master_port=6060 run_pretraining_multimae.py \
--in_domains rgb-him-eim --out_domains rgb --model pretrain_multimae_small \
--alphas 8.0 1.0 1.0 --mask_strategy one --num_encoded_tokens 98 --batch_size 78


# DINO
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 run_pretraining_dino.py \
--arch vit_small --batch_size_per_gpu 78


# TRAIN FROM SCRATCH
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --master_port=6060 supervised.py \
-no_pre_HE --model multivit_small --batch_size 78
```
### Modifying configs
The training scripts support both YAML config files and command-line arguments. See [here](cfgs/pretrain) for pre-training config files.

To modify pre-training settings, either edit / add config files or provide additional command-line arguments.

For a list of possible arguments, see [`run_pretraining_multimae.py`](run_pretraining_multimae.py).

:information_source: Config files arguments override default arguments, and command-line arguments override both default arguments and config arguments.
