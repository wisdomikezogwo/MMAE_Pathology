# Fine-tuning and K-NN

We provide fine-tuning scripts for classification.

- [General information](#general-information)
- [Classification](#classification)

## General information

### Loading pre-trained models

All our fine-tuning scripts support models in the MultiMAE / MultiViT format. Pre-trained models using the timm / ViT format can be converted to this format using the [`vit2multimae_converter.py`](tools/vit2multimae_converter.py)
 script.

### Modifying configs
The training scripts support both YAML config files and command-line arguments. See [here](cfgs/finetune) for all fine-tuning config files.

To modify fine-training settings, either edit / add config files or provide additional command-line arguments.

:information_source: Config files arguments override default arguments, and command-line arguments override both default arguments and config arguments.

## Classification

We use 4 A4000 GPUs for classification fine-tuning. Configs can be found [here](cfgs/finetune/cls).

To fine-tune MultiMAE on NCT-CRC-HE-100K classification using default settings, run:

```bash
KNN
# MMAE 
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --master_port=6061 run_knn.py \
--pretrained_weights ./output_dir/pretrained_experiment_name/checkpoint-XXX.pth --model multivit_small \
--batch_size 128 --linprobe_n 10000 --dump_or_load dump --nb_knn 10 20 --use_cuda False

#DINO
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --master_port=6061 run_knn.py \
--pretrained_weights ./output_dir/pretrained_experiment_name/checkpoint-XXX.pth --model vit_small --batch_size 128 \
--linprobe_n 10000 --dump_or_load dump --nb_knn 10 20 --use_cuda False

Finetune
# MMAE
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --master_port=6061 run_mmae_finetune.py \
-no_pre_HE --finetune ./output_dir/pretrained_experiment_name/checkpoint-XXX.pth --model multivit_small \
--batch_size 25 --linprobe_n 100 1000

#DINO
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --master_port=6061 run_dino_finetune.py \
--finetune ./output_dir/pretrained_experiment_name/checkpoint-XXX.pth --model vit_small \
--arch vit_small --batch_size 25 --linprobe_n 100 1000
```


- For a list of possible arguments, see [`run_finetuning_cls.py`](run_mmae_finetune.py).
