# RedeFISH
Automatic Cell Alignment in Ultra-large Spatial and Single-cell Transcriptomics Data

![1A](https://github.com/Roshan1992/RedeFISH/assets/11591480/f723d5c4-05b5-4211-8ad3-5a50d6f31f83)



## Overview

RedeFISH is an automatic tool for cell alignment in imaging-based spatial transcriptomics (ST) and scRNA-seq data through deep reinforcement learning. This method aims to identify functional-defined cells in ST data that exhibit the highest degree of expression similarity with cells in scRNA-seq data. Through the incorporation of scRNA-seq data, this method additionally undertakes the task of inferring whole-transcriptome expression profiles for the aforementioned identified cells. RedeFISH is a python package written in Python 3.9 and pytorch 1.12. It allows GPU to accelerate computational efficiency.


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

## Quick Start

### Step 1. Prepare input files

RedeFISH requires 2 file as input:

__[1] A csv file for single-cell ST data:__ This file must includes at least 3 columns, namely __x__, __y__ and corresponding __gene__.

![image](https://user-images.githubusercontent.com/11591480/236604144-21a769c2-398b-40e2-9dc7-084d7630241d.png)

__[2] An Anndata h5ad file for scRNA-seq data:__ This file must includes expression matrix and cell type annotation.

![image](https://user-images.githubusercontent.com/11591480/236605176-6551c703-e19b-42f0-9c43-4022e41b7eb4.png)

### Step 2. Implement RedeFISH

See <a href="https://github.com/Roshan1992/Redesics/blob/main/example.ipynb" target="_blank">example</a> for implementing RedeFISH on imaging-based single-cell ST platforms.

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

__[1] cell_expression.h5ad:__ Expression matrix of identified cells.

__[2] cell_expression.predict.h5ad:__ Whole-transcriptome expression profiles of identified cells through imputation.

__[3] cell_feature.csv:__ A csv file includes features of identified cells (cell center coordinates, girth, area, roundness, mRNA_count, cell type by label transferring). The index refers to cell IDs.

__[4] transcripts_classification.csv:__ The result for transcript assignments. It includes transcript coordinates (column: x and y) and corresponding cell IDs (column: mRNA_class). Positive interger in __mRNA_class__ column refer to non-noise assignment, while zero indicates transcript classified as background noise.

__[5] cell_boundary.h5:__ A h5 file contains polygonal regions of cell boundary for identified cells.





