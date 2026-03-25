# fvcCOVER 
**_Code for image processing, build reference/annotation data and semantic segmentation modelling for mapping fractional vegetation cover in UAS RGB and multispectral imagery._**

Sotomayor, L.N., *et al.* (2025). **Mapping fractional vegetation cover in UAS RGB and multispectral imagery in semi-arid Australian ecosystems using CNN-based semantic segmentation**.  
*Landscape Ecology*, 40, 169. https://doi.org/10.1007/s10980-025-02193-y

<!-- ![fvcCover](https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/fvc_mapping_predictions.png) -->
<img src="https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/fvc_mapping_predictions.png" width="500">

## CNN-based workflow for FVC mapping application:

<img src="https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/cnn_workflow_sites.png" width="400">
<!-- ![mutlispectralMultipleClasses](https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/cnn_workflow_sites.png) -->


## Installation

```diff
#rebuild environment with dependencies 
install miniconda (not anaconda)
install micromamba as a standalone CLI tool to manage Conda-compatible environments and packages
micromamba create -f environment.yml
```

## Train (site-specific U-Net)

Training for the site-specific models is driven by:

- `phase_3_models/unet_site_models/main_site_specific_models.py`

Before running, update paths and training settings in:

- `phase_3_models/unet_site_models/config_param.py` (data folders, class labels, number of classes, checkpoint dir, device, epochs, etc.)
- `phase_3_models/unet_site_models/main_site_specific_models.py` (default output/log directories are currently hard-coded)

Run from the repo root:

```bash
python phase_3_models/unet_site_models/main_site_specific_models.py
```

## Validation / test metrics (training)

When training site-specific U-Net models via `phase_3_models/unet_site_models/main_site_specific_models.py`, metrics are written to the training `output_dir` defined inside that script.

Outputs you can use for **validation** and **test** reporting include:

- Per-block metrics (text): `final_model_metrics_block_*.txt` and `best_model_metrics_block_*.txt`
- Per-block metrics (JSON): `block_*_val_metrics.json` and `block_*_best_val_metrics.json`
- Loss curves: `loss_metrics.txt` and `average_training_validation_loss_plot_across_blocks.png`
- Confusion matrices / aggregated summaries are also written under the same `output_dir` (filenames depend on the evaluator utilities).

Checkpoints (best model per block) are saved under `config_param.CHECKPOINT_DIR`.

If you enabled TensorBoard logging, the log directory is set in `setup_logging_and_checkpoints()` inside `main_site_specific_models.py` and can be viewed with:

```bash
tensorboard --logdir <tb_logs_path>
```

## Inference workflow (tiles 22 / 55)

Step-by-step commands to run inference, generate COG + GeoJSON + thumbnails, and build STAC metadata are in:

- [INFERENCE_WORKFLOW.md](INFERENCE_WORKFLOW.md)

## Web viewer (COG + GeoJSON)

The viewer in `cnn_mappingAI_viewer.html` loads a Cloud-Optimized GeoTIFF (COG) using HTTP `Range` requests (206 Partial Content), so you need a Range-capable server.

Local (from repo root):

```bash
python3 bin/range_http_server.py 8001
```

Docker (reproducible):

```bash
docker compose up --build viewer
```

Open:

- http://127.0.0.1:8001/cnn_mappingAI_viewer.html

If port `8001` is already in use, change the left side of the port mapping in `docker-compose.yml` (e.g. `8002:8001`).

## U-Net setup + parameter counts

Parameter counts below were computed by loading each checkpoint into the inference U-Net and summing `model.parameters()` ("authoritative").

Command used:

```bash
python phase_3_models/unet_site_models/count_checkpoint_params.py --model-path \
  "phase_3_models/unet_site_models/outputs_ecosystems/low/original/block_3_epoch_108.pth" \
  "phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth" \
  "phase_3_models/unet_site_models/outputs_ecosystems/dense/original/block_3_epoch_105.pth"
```

Model setup (all checkpoints): Standard U-Net, depth 5, features `[64, 128, 256, 512, 1024]`, input bands = 5.

| Ecosystem | Checkpoint | Output classes | Total params | final_conv params | Rest params | Buffers (BN stats) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Low | [phase_3_models/unet_site_models/outputs_ecosystems/low/original/block_3_epoch_108.pth](phase_3_models/unet_site_models/outputs_ecosystems/low/original/block_3_epoch_108.pth) | 3 | 124,375,491 | 195 | 124,375,296 | 24,086 |
| Medium | [phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth](phase_3_models/unet_site_models/outputs_ecosystems/medium/original/block_2_epoch_55.pth) | 4 | 124,375,556 | 260 | 124,375,296 | 24,086 |
| Dense | [phase_3_models/unet_site_models/outputs_ecosystems/dense/original/block_3_epoch_105.pth](phase_3_models/unet_site_models/outputs_ecosystems/dense/original/block_3_epoch_105.pth) | 5 | 124,375,621 | 325 | 124,375,296 | 24,086 |


## Dataset available
- **You can find the whole raw dataset used for phase B** in workflow: [![DOI](https://zenodo.org/badge/DOI/110.5281/zenodo.15036860.svg)](https://doi.org/10.5281/zenodo.15036860)

Sotomayor, L. N., Megan, L., Krishna, L., Sophia, H., Molly, M., & Arko, L. (2025). Fractional Vegetation Cover Mapping - UAS RGB and Multispectral Imagery, CNN algorithms, Semi-Arid Australian Ecosystems Coverage [Data set]. Zenodo.


- **You can find a sample for the reference dataset and CNN modelling purpose for phase C**:

 [![DOI](https://zenodo.org/badge/DOI/10.6084/m9.figshare.27776145.v1.svg)](https://doi.org/10.6084/m9.figshare.27776145.v1)  
  Sotomayor, Laura (2024). Low vegetation site. figshare. Dataset.  
  <!-- DOI: [10.6084/m9.figshare.27776145.v1](https://doi.org/10.6084/m9.figshare.27776145.v1) -->

 [![DOI](https://zenodo.org/badge/DOI/10.6084/m9.figshare.27871806.v1.svg)](https://doi.org/10.6084/m9.figshare.27871806.v1)  
  Sotomayor, Laura (2024). Medium vegetation site. figshare. Dataset.  
  <!-- DOI: [10.6084/m9.figshare.27871806.v1](https://doi.org/10.6084/m9.figshare.27871806.v1) -->

 [![DOI](https://zenodo.org/badge/DOI/10.6084/m9.figshare.27871893.v1.svg)](https://doi.org/10.6084/m9.figshare.27871893.v1)  
  Sotomayor, Laura (2024). Dense vegetation site. figshare. Dataset.  
  <!-- DOI: [10.6084/m9.figshare.27871893.v1](https://doi.org/10.6084/m9.figshare.27871893.v1) -->

<!-- [FVC classes based on growth form and structure ](https://figshare.com/projects/Reference_data_for_semi-arid_environments/227859) -->

## Cite code for fvcCOVER
This code can be cited and downloaded from: [![DOI](https://zenodo.org/badge/DOI/110.5281/zenodo.15036626.svg)](https://doi.org/10.5281/zenodo.15036626)

Sotomayor, L. N. (2025). fvcCOVER: Code for image processing, build reference/annotation data and semantic segmentation modelling for mapping fractional vegetation cover in UAS RGB and multispectral imagery. Zenodo.

## Acknowledgments
- **Orthomosaics from drone imagery**: the raw RGB (1 cm) and multispectral (5 cm) orthomosaics at **phase A** in workflow can be found:
TERN Landscapes, TERN Surveillance Monitoring, Stenson, M., Sparrow, B., & Lucieer, A. (2022).
Drone RGB and Multispectral Imagery from TERN plots across Australia. Version 1. Terrestrial Ecosystem Research Network. Dataset. 
[Access TERN drone RGB and Multispectral orthomosaics here](https://portal.tern.org.au/metadata/TERN/39de90f5-49e3-4567-917c-cf3e3bc93086).
<!-- <img src="https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/orthomosaic_sites.png" width="500"> -->
<figure style="text-align:center;">
  <img src="https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/orthomosaic_sites.png" width="500">
  <figcaption style="font-size:90%;"><b>Figure.</b> Resampled 1 cm multispectral orthomosaics (<b>Phase A</b>) across vegetation types used in modelling CNN workflow</figcaption>
</figure>
<br><br>

- **Contribution for reference/labelling dataset process**: we would like to acknowledge and thank all the individuals who contributed to the labelling process:

Prof. Megan Lewis (School of Biological Sciences, University of Adelaide), 
Dr Krishna Lamsal (School of Geography, Planning, and Spatial Sciences, UTAS), 
Sophia Hoyer (School of Geography, Planning, and Spatial Sciences, UTAS) and
Molly Marshall (School of Geography, Planning, and Spatial Sciences, UTAS).

<!-- ### Check code: Paper2/Chap3
[LiDAR 3D Voxel Automation to identify trees, shrubs, and grasses using deep learning-based computer vision applications ](https://github.com/LNSOTOM/ecosystem_structure) -->

