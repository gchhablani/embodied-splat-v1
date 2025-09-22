# EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device

Repository for the paper [EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](https://gchhablani.github.io/embodied-splat-web).

This repository contains code corresponding to the paper accepted at ICCV 2025.

**Disclaimer**: Please note that using a different kind (or version) of mesh reconstruction strategies may lead to different results than those reported in the paper.

Note that the names for the scenes in this repository differ from the names in the paper. This is because the names were changed to anonymize the scenes for the review process.

The mapping is as follows:

- `lounge` -> `grad_lounge`
- `classroom` -> `clough_classroom`
- `conf_a` -> `piedmont`
- `conf_b` -> `castleberry`
- `conf_c` -> `coda_conference_room`

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/gchhablani/embodied-splat.git
   ```

2. Create the conda environment from the provided YAML file:

   ```bash
   conda env create -f embodied_splat.yml
   ```

3. Install Habitat-Lab and Habitat-Sim. We use Habitat-Sim version `0.3.1` and Habitat-Lab version `0.3.1`. Please follow the instructions in the [Habitat-Lab repository](https://github.com/facebookresearch/habitat-lab) for installation. For habitat-sim, we recommend using the conda installation, while for habitat-lab, we recommend installing from source. Install both `habitat-lab` and `habitat-baselines` in editable mode.

4. Install the `embodied_splat` package in editable mode:

   ```bash
   pip install -e .
   ```

5. Download the data and checkpoints using the provided script:

   ```bash
   ./download_data_and_ckpts.sh
   ```

   Specifically, this will download the following datasets:
   - MuSHRoom scenes with DN-Splatter meshes
   - Polycam scenes with DN-Splatter meshes
   - Polycam scenes with Polycam meshes

   It will also download the episode datasets for the above, and also the episode datasets for HM3D and HSSD used for pre-training.
   Finally, it will download the pre-trained checkpoints for the models used in the paper.

   For downloading the HM3D and HSSD scenes, please following instructions from the official [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) repository.

6. Activate the conda environment:

   ```bash
   conda activate embodied_splat
   ```

7. Replace the Weights & Biases username and project across the script of choice. Currently, they are set as follows:

   ```bash
   WB_ENTITY="user"
   PROJECT_NAME="embodied_splat"
   ```

   You may need to create a weights & biases account if you don't have one already, and log in using the CLI. You may also choose to use tensorboard, but this will require modifying the template scripts.
   Please create an issue if you need help with this step.

8. Run any of the training scripts to ensure everything is set up correctly.
    **You may have to modify or create your own scripts based on your cluster setup**. The scripts are written for a SLURM-based cluster with specific GPU types and memory requirements. You can find the scripts in the `scripts/` directory.

    ```bash
    ./scripts/train_v2/submit_imagenav_mushroom_dn_splatter.sh
    ```

## Scene Reconstruction

The scene reconstructions used in this paper are built using the [DN-Splatter](https://github.com/maturk/dn-splatter) repository. Note that at the time of this writing, the repository has evolved and updated.
For reference, we use the following version of the repository (with some added depth and normal encoders): <https://github.com/gchhablani/dn-splatter/tree/embodied_splat>.
Please create an issue on either of these repositories in case you have trouble setting or running the mesh recontrsuction, or if something is missing from this fork.

## Training

### Pre-training on HM3D and HSSD

#### HM3D

```bash
./scripts/train_v2/submit_imagenav_hm3d.sh
```

#### HSSD

```bash
./scripts/train_v2/submit_imagenav_hssd_high_lr.sh
```

### Fine-tuning on MuSHRoom scenes

#### HM3D on DN-Splatter Meshes

```bash
./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter_mushroom.sh
./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter_mushroom_v2.sh # for more scenes
```

#### HSSD on DN-Splatter Meshes

```bash
./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter_mushroom_hssd.sh
```

### Fine-tuning on Polycam scenes

#### HM3D on DN-Splatter Meshes

```bash
./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter.sh
./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter_2.sh # for more scenes
```

#### HM3D on Polycam Meshes

```bash
./scripts/fine_train_v2/submit_imagenav_polycam_mesh.sh
./scripts/fine_train_v2/submit_imagenav_polycam_mesh_2.sh # for more scenes
```

#### HSSD on DN-Splatter Meshes

```bash
./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter_hssd.sh
```

#### HSSD on Polycam Meshes

```bash
./scripts/fine_train_v2/submit_imagenav_polycam_mesh_hssd.sh
```

## Evaluation

### Zero-shot evaluation of HM3D and HSSD pre-trained models

#### HM3D on individual Polycam DN-Splatter Meshes

```bash
./scripts/eval_v2/cross_evals/submit_hm3d_on_dn_splatter.sh
./scripts/eval_v2/cross_evals/submit_hm3d_on_dn_splatter_2.sh # for more scenes
```

#### HM3D on individual Polycam Meshes

```bash
./scripts/eval_v2/cross_evals/submit_hm3d_on_polycam
./scripts/eval_v2/cross_evals/submit_hm3d_on_polycam_2.sh # for more scenes
```

#### HM3D on MuSHRoom DN-Splatter Meshes

```bash
./scripts/eval_v2/zero_shot_evals/submit_hm3d_on_dn_splatter_mushroom.sh
./scripts/eval_v2/zero_shot_evals/submit_hm3d_on_dn_splatter_mushroom_v2.sh # for more scenes
```

#### HSSD on individual Polycam DN-Splatter Meshes

```bash
./scripts/eval_v2/cross_evals/submit_hssd_on_dn_splatter.sh
```

#### HSSD on individual Polycam Meshes

```bash
./scripts/eval_v2/cross_evals/submit_hssd_on_polycam_mesh.sh
./scripts/eval_v2/cross_evals/submit_hssd_on_polycam_mesh_2.sh # for more scenes
```

#### HSSD on MuSHRoom DN-Splatter Meshes

```bash
./scripts/eval_v2/zero_shot_evals/submit_hssd_on_dn_splatter_mushroom.sh
./scripts/eval_v2/zero_shot_evals/submit_hssd_on_dn_splatter_mushroom_v2.sh # for more scenes
```

### Fine-tuned model evaluation

#### Fine-tuned HM3D on MuSHRoom DN-Splatter Meshes

```bash
./scripts/fine_train_eval_v2/submit_imagenav_polycam_dn_splatter_mushroom.sh
./scripts/fine_train_eval_v2/submit_imagenav_polycam_dn_splatter_mushroom_v2.sh # for more scenes
```

#### Fine-tuned HSSD on MuSHRoom DN-Splatter Meshes

```bash
./scripts/fine_train_eval_v2/submit_imagenav_polycam_dn_splatter_mushroom_hssd.sh
```

#### Fine-tuned HM3D on individual Polycam DN-Splatter Meshes

```bash
./scripts/fine_train_eval_v2/submit_imagenav_polycam_dn_splatter.sh
./scripts/fine_train_eval_v2/submit_imagenav_polycam_dn_splatter_2.sh # for more scenes
```

#### Fine-tuned HSSD on individual Polycam DN-Splatter Meshes

```bash
./scripts/fine_train_eval_v2/submit_imagenav_polycam_dn_splatter_hssd.sh
```

#### Fine-tuned HM3D on individual Polycam Meshes

```bash
./scripts/fine_train_eval_v2/submit_imagenav_polycam_mesh.sh
./scripts/fine_train_eval_v2/submit_imagenav_polycam_mesh_2.sh # for more scenes
```

#### Fine-tuned HSSD on individual Polycam Meshes

```bash
./scripts/fine_train_eval_v2/submit_imagenav_polycam_mesh_hssd.sh
```

### Other Miscenallaneous Training and Evaluation scripts

There are other scripts for training and evaluation in the `scripts/` directory. Please refer to the script parameters to understand the purpose of scripts. These scripts were used for ablation studies and other experiments. For example `./scripts/eval_v2/zero_shot_evals/` contains scripts to run average performance across different training steps for HM3D and HSSD pre-training. Similarly, `./scripts/data` contains scripts to create the PointNav episodes. There are scripts which evaluate post fine-tuning performance on the HM3D and HSSD validation sets to ensure that the performance does not deteriorate. There are also scripts with `of` in the names used for overfitting evals on single scenes, and scripts in `train_v2` which deal with overfitting. For brevity of this README, we do not list all of them here. Please create an issue if you need help with any of these scripts.

## Real-world deployment

**Coming soon!**
