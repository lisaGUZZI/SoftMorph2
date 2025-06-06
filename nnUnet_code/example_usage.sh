# !bin/bash


# Activate conda environment
conda activate softmorph

# Set nnUNet paths
export nnUNet_results=".../nnUNet_results"
export nnUNet_raw=".../nnUNet_raw"
export nnUNet_preprocessed=".../nnUNet_preprocessed"
export nnUNet_compile='f'

# Specify the operation
export SOFTMORPH_OP="closing"


nnUNetv2_train 7 3d_fullres 0 
# nnUNetv2_train 7 3d_fullres 1 
# nnUNetv2_train 7 3d_fullres 2 
# nnUNetv2_train 7 3d_fullres 3 
# nnUNetv2_train 7 3d_fullres 4 