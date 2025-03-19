# fvcCOVER 
**_Code for image processing, build reference/annotation data and semantic segmentation modelling for mapping fractional vegetation cover in UAS RGB and multispectral imagery._**

<!-- ![fvcCover](https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/fvc_mapping_predictions.png) -->
<img src="https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/fvc_mapping_predictions.png" width="500">

## CNN-based workflow for FVC mapping application:

<img src="https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/cnn_workflow_sites.png" width="400">
<!-- ![mutlispectralMultipleClasses](https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/cnn_workflow_sites.png) -->


## Installation

```diff
#rebuild environment with dependencies 
install miniconda (not anaconda)
conda install -c conda-forge mamba 
mamba env create --file environment.yml
```

## Dataset available
- **You can find the whole raw dataset used for phase B** in workflow: [![DOI](https://zenodo.org/badge/DOI/110.5281/zenodo.15036860.svg)](https://doi.org/10.5281/zenodo.15036860)

Sotomayor, L. N. (2025). Fractional Vegetation Cover Mapping - UAS RGB and Multispectral Imagery, CNN algorithms, Semi-Arid Australian Ecosystems Coverage [Data set]. Zenodo.


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

### Method
Coming Paper in Peer Review titled: 'Mapping fractional vegetation cover in UAS RGB and multispectral imagery in semi-arid Australian ecosystems using CNN-based semantic segmentation'.

## Acknowledgments
- **Orthomosaics from drone imagery**: the RGB (1 cm) and multispectral (5 cm) orthomosaics at **phase A** in workflow can be found:
<img src="https://github.com/LNSOTOM/fvc_composition/blob/main/phase_1_image_processing/img/orthomosaic_sites.png" width="500">

TERN Landscapes, TERN Surveillance Monitoring, Stenson, M., Sparrow, B., & Lucieer, A. (2022).
Drone RGB and Multispectral Imagery from TERN plots across Australia. Version 1. Terrestrial Ecosystem Research Network. Dataset. 
[Access TERN drone RGB and Multispectral orthomosaics here](https://portal.tern.org.au/metadata/TERN/39de90f5-49e3-4567-917c-cf3e3bc93086).

- **Contribution for reference/labelling dataset process**: we would like to acknowledge and thank all the individuals who contributed to the labelling process:

Prof. Megan Lewis (School of Biological Sciences, University of Adelaide), 
Dr Krishna Lamsal (School of Geography, Planning, and Spatial Sciences, UTAS), 
Sophia Hoyer (School of Geography, Planning, and Spatial Sciences, UTAS),
Molly Marshall (School of Geography, Planning, and Spatial Sciences, UTAS),
Dr Agustina Barros (Researcher at National Scientific and Technical Research Council) and
Dr Sebastian Rossi (Researcher at National Scientific and Technical Research Council).

<!-- ### Check code: Paper2/Chap3
[LiDAR 3D Voxel Automation to identify trees, shrubs, and grasses using deep learning-based computer vision applications ](https://github.com/LNSOTOM/ecosystem_structure) -->

