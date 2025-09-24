# EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device

This repository accompanies the paper  
[**EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device**](https://gchhablani.github.io/embodied-splat), accepted at **ICCV 2025**.

---

## 📌 Notes

- **Mesh reconstruction disclaimer**: Results may vary if different mesh reconstruction strategies (or versions) are used compared to those in the paper.  
- **Scene naming**: Scene names in this repository differ from the paper due to anonymization during review. Mapping is as follows:

  - `lounge` → `grad_lounge`  
  - `classroom` → `clough_classroom`  
  - `conf_a` → `piedmont`  
  - `conf_b` → `castleberry`  
  - `conf_c` → `coda_conference_room`

---

## ⚙️ Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/gchhablani/embodied-splat-v1.git
   ```

2. **Create the conda environment**

   ```bash
   conda env create -f embodied_splat.yml
   ```

3. **Install Habitat-Lab and Habitat-Sim**  
   - Habitat-Sim: version `0.3.1` (recommend **conda install**)  
   - Habitat-Lab: version `0.3.1` (recommend **source install**)  
   - Install both `habitat-lab` and `habitat-baselines` in editable mode.  
   See [Habitat-Lab installation guide](https://github.com/facebookresearch/habitat-lab).

4. **Install EmbodiedSplat in editable mode**

   ```bash
   pip install -e .
   ```

5. **Download data and checkpoints**

   ```bash
   ./download_data_and_ckpts.sh
   ```

   This script downloads:
   - MuSHRoom scenes with DN-Splatter meshes  
   - Polycam scenes with DN-Splatter and Polycam meshes  
   - Episode datasets (MuSHRoom, Polycam, HM3D, HSSD)  
   - Pre-trained model checkpoints  

   ⚠️ For HM3D and HSSD scenes, follow the official [Habitat-Lab dataset instructions](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).

6. **Activate the environment**

   ```bash
   conda activate embodied_splat
   ```

7. **Configure logging**  
   Replace default Weights & Biases config in scripts:

   ```bash
   WB_ENTITY="user"
   PROJECT_NAME="embodied_splat"
   ```

   - Requires a [Weights & Biases](https://wandb.ai) account and CLI login.  
   - Alternatively, you may switch to TensorBoard (requires script edits).  
   - Create an issue if you need help.

8. **Test the installation**  
   Run a sample training script (adapt as needed for your cluster). Scripts are designed for **SLURM clusters** with specific GPU/memory settings.  

   ```bash
   ./scripts/train_v2/submit_imagenav_mushroom_dn_splatter.sh
   ```

---

## 🏗️ Scene Reconstruction

Reconstruction is performed with [**DN-Splatter**](https://github.com/maturk/dn-splatter).  
For reproducibility, we used this fork with added depth/normal encoders:  
<https://github.com/gchhablani/dn-splatter/tree/embodied_splat>.  

If you encounter issues with reconstruction, please open an issue in either repository.

---

## 🔧 Preprocessing

- **HSSD**
  ```bash
  ./scripts/data/v0.5/create_pointnav_hssd.sh
  ```
- **[MuSHRoom](https://xuqianren.github.io/publications/MuSHRoom/) Dataset**
  ```bash
  ./scripts/data/v0.5/create_pointnav_mushroom.sh
  ```
- **DN-Splatter meshes**
  ```bash
  ./scripts/data/v0.5/create_pointnav_dn_splatter.sh
  ```
- **Polycam meshes**
  ```bash
  ./scripts/data/v0.5/create_pointnav_polycam_mesh.sh
  ```

## 🏋️ Training

### Pre-training

- **HM3D**

  ```bash
  ./scripts/train_v2/submit_imagenav_hm3d.sh
  ```

- **HSSD**

  ```bash
  ./scripts/train_v2/submit_imagenav_hssd_high_lr.sh
  ```

### Fine-tuning on MuSHRoom

- **HM3D (DN-Splatter meshes)**

  ```bash
  ./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter_mushroom.sh
  ./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter_mushroom_v2.sh
  ```

- **HSSD (DN-Splatter meshes)**

  ```bash
  ./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter_mushroom_hssd.sh
  ```

### Fine-tuning on Polycam

- **HM3D (DN-Splatter meshes)**

  ```bash
  ./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter.sh
  ./scripts/fine_train_v2/submit_imagenav_polycam_dn_splatter_2.sh
  ```

- **HM3D (Polycam meshes)**

  ```bash
  ./scripts/fine_train_v2/submit_imagenav_polycam_mesh.sh
  ./scripts/fine_train_v2/submit_imagenav_polycam_mesh_2.sh
  ```

- **HSSD (DN-Splatter & Polycam meshes)**  
  Scripts available in `fine_train_v2/`.

---

## 📊 Evaluation

### Zero-shot Evaluation

- HM3D / HSSD on Polycam and MuSHRoom (DN-Splatter + Polycam meshes)  
  Example:

  ```bash
  ./scripts/eval_v2/cross_evals/submit_hm3d_on_dn_splatter.sh
  ```

### Fine-tuned Model Evaluation

- Scripts in `fine_train_eval_v2/` for HM3D/HSSD across MuSHRoom and Polycam meshes.
Example:

  ```bash
  ./scripts/fine_train_eval_v2/submit_imagenav_polycam_dn_splatter_mushroom.sh
  ```

## Miscellaneous

- Ablations, overfitting tests, and intermediate evaluations are in `scripts/`.
- Example: `eval_v2/zero_shot_evals/` for step-wise evaluation of pre-trained models on different types of meshes (averaged).  
- `scripts/data` contains the scripts to generate the PointNav datasets.
- Some evaluation scripts with the name ending with `hm3d_on_hm3d` and `hssd_on_hssd` are for evaluating fine-tuned models on different meshes on their respective pre-training evaluation sets to see how well they perform on the original datasets.
- Open an issue if you need guidance for specific scripts.

---

## 🌍 Real-world Deployment
We build upon the [home-robot](https://github.com/facebookresearch/home-robot) framework for real-world deployment.

**Details coming soon!**

---

## 🙋 Support

If you encounter issues, please open a GitHub issue in this repository.
