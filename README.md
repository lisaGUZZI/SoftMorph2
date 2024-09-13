# SoftMorph : Differentiable Probabilistic Morphological Operators for Image Analysis

* [Content](#content)
* [Usage](#usage)
* [Contact](#contact)

## Content
This repository contains the code for probabilistic and differentiable morphological filters. 
* `SoftMorph2D.py` : 2D operations for Erosion, Dilation, Opening, Closing and Skeletonization
* `SoftMorph3D.py` : 3D operations for Erosion, Dilation, Opening, Closing and Skeletonization
* `butterfly_segmentations.zip` : Corrected segmentation masks for the Butterfly dataset from 

## Usage
### Input image format
Input images should contain values in the range [0, 1]. Supported dimensions are :
* **2D** images of shape [batch_size, channels, heigth, width] or [height, width]
* **3D** images of shape [batch_size, channels, depth, heigth, width] or [depth, height, width]

### Application
The filters have been tested on  :
* Binary and probabilistic values ranging between [0, 1]
* In loss function
* As the final layer of a segmentation network

### Filters description
* ***SoftErosion*** : Erode the foreground 
    * <u>forward parameters</u> (*image* = input image, *iterations* = number of times the morphological operation is repeated, *connectivity* = structuring element [4, 8] in 2D and [6, 18, 26] in 3D)
* ***SoftDilation*** : Dilate the foreground
    * <u>forward parameters</u> (*image* = input image, *iterations* = number of times the morphological operation is repeated, *connectivity* = structuring element [4, 8] in 2D and [6, 18, 26] in 3D)
* ***SoftClosing*** : Dilation followed by an Erosion
    * <u>forward parameters</u> (*image* = input image, *iterations* = number of times each morphological operation is repeated, *dilation_connectivity* = structuring element [4, 8] in 2D and [6, 18, 26] in 3D for dilation operation, *erosion_connectivity* = structuring element for erosion)
* ***SoftOpening*** : Erosion followed by a Dilation
    * <u>forward parameters</u> (*image* = input image, *iterations* = number of times each morphological operation is repeated, *dilation_connectivity* = structuring element [4, 8] in 2D and [6, 18, 26] in 3D for dilation operation, *erosion_connectivity* = structuring element for erosion)
* ***SoftSkeletonizer*** : Repeated thinning operation to extract the centerline of the foreground
    * <u>init parameters</u> (*max_iter* = number of repeated thinning operation)
    * <u>forward parameters</u> (*image* = input image)



## Contact
For questions or inquiries, please contact Lisa Guzzi at lisa.guzzi@inria.fr 
