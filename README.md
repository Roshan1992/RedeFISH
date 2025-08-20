# RedeFISH
Cell Segmentation as Strategic Decision Making

(![Figure 1](https://github.com/user-attachments/assets/8ba4e0be-d7bd-44c0-bc6c-ed348a65e9dc)



## Overview

RedeFISH is a reinforcement learning-based tool for cell segmentation and gene imputation in imaging-based spatial transcriptomics (ST). It leverages scRNA-seq to guide transcript assignment and cell boundary delineation, learning an optimal segmentation strategy directly from ST data without stained images, thereby ensuring robustness to staining variability and tissue heterogeneity. RedeFISH is a python package written in Python 3.9 and pytorch 1.12. It allows GPU to accelerate computational efficiency.


## Installation

[1] Install <a href="https://www.anaconda.com/" target="_blank">Anaconda</a> if not already available

[2] Clone this repository:
```
    git clone https://github.com/Roshan1992/RedeFISH.git
```

[3] Change to RedeFISH directory:
```
    cd RedeFISH
```

[4] Create a conda environment with the required dependencies:
```
    conda env create -f environment.yml
```

[5] Activate the RedeFISH_env environment you just created:
```
    conda activate RedeFISH_env
```

[6] Install RedeFISH:
```
    pip install .
```

[7] Install pytorch:

If GPU available (https://pytorch.org/get-started/previous-versions/):
```
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
If GPU not available:
```
    pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
```
Note: ensure that the corresponding PyTorch installation command matches your system's CUDA version

## Quick Start

### Step 1. Prepare input files

RedeFISH requires 2 file as input:

__[1] A csv file for single-cell ST data:__ This file must includes at least 3 columns, namely __x__, __y__ and corresponding __gene__.

![image](https://user-images.githubusercontent.com/11591480/236604144-21a769c2-398b-40e2-9dc7-084d7630241d.png)

__[2] An Anndata h5ad file for scRNA-seq data:__ This file must includes expression matrix and cell type annotation.

![image](https://user-images.githubusercontent.com/11591480/236605176-6551c703-e19b-42f0-9c43-4022e41b7eb4.png)

Click <a href="https://drive.google.com/file/d/1_t5C9_1f0084w-iIAuz_xBUvNpp1vn2j/view?usp=drive_link" target="_blank">here</a> to access the Mouse Ileum dataset, including spatial transcriptomics and single-cell example data

### Step 2. Implement RedeFISH

See <a href="https://github.com/Roshan1992/Redesics/blob/main/example.ipynb" target="_blank">example</a> for implementing RedeFISH on imaging-based single-cell ST platforms.

See <a href="https://github.com/Roshan1992/Redesics/blob/main/example_for_Stereo_seq.ipynb" target="_blank">example_for_Stereo_seq</a> for implementing RedeFISH on Stereo-seq platforms.

### Step 3. Output

The contents of the output directory in tree format will be displayed as described below:

```
    Output PATH
    ├── cell_boundary.h5
    ├── cell_expression.h5ad
    ├── cell_expression.predict.h5ad
    ├── cell_feature.csv
    └── transcripts_classification.csv
```

__[1] cell_expression.h5ad:__ Expression matrix of segmented cells.

__[2] cell_expression.predict.h5ad:__ Whole-transcriptome expression profiles of segmented cells through imputation.

__[3] cell_feature.csv:__ A csv file includes features of segmented cells (cell center coordinates, girth, area, roundness, mRNA_count, cell type by label transferring). The index refers to cell IDs.

__[4] transcripts_classification.csv:__ The result for transcript assignments. It includes transcript coordinates (column: x and y) and corresponding cell IDs (column: mRNA_class). Positive interger in __mRNA_class__ column refer to non-noise assignment, while zero indicates transcript classified as background noise.

__[5] cell_boundary.h5:__ A h5 file contains polygonal regions of cell boundary for segmented cells.





