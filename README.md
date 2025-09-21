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

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/gchhablani/embodied-splat.git
   ```

2. Create the conda environment from the provided YAML file:

   ```bash
   conda env create -f embodied_splat.yml
   ```

3. Install Habitat-Lab and Habitat-Sim following the instructions below:

4. Install the package in editable mode:

   ```bash
   pip install -e .
   ```

5. Download the data and place it in the `data/` directory. The exact structure should look like:

    ```
    TODO: Add data structure here
    ```

    For downloading the custom scenes and corresponding episodes, please check out our HuggingFace repository.
    For downloading the HM3D and HSSD scene datasets, please folow the instructions in the [Habitat-Lab repository]()

5. Download the pre-trained HM3D and HSSD checkpoints and place them as follows:
   TODO:
6. Activate the conda environment:

   ```bash
   conda activate embodied_splat
   ```

7. Run any of the training scripts to ensure everything is set up correctly.
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

### Fine-tuning on MuSHRoom scenes

### Fine-tuning on Polycam scenes with DN-Splatter Meshes

### Fine-tuning on Polycam scenes with Polycam Meshes

## Evaluation

### Zero-shot evaluation of HM3D and HSSD pre-trained models

### Fine-tuned model evaluation

## Real-world deployment

## Citation

If you find this work useful in your research, please consider citing:

```
TODO
```
