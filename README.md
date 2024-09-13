# SoftMorph : Differentiable Probabilistic Morphological Operators for Image Analysis

* [Content](#content)
* [Usage](#usage)
* [Contact](#contact)

## Content
This repository contains the code for probabilistic and differentiable morphological filters. 
* `SoftMorph_2D.py` : 2D operations for Erosion, Dilation, Opening, Closing and Skeletonization
* `SoftMorph_3D.py` : 3D operations for Erosion, Dilation, Opening, Closing and Skeletonization
* `butterfly_segmentations.zip` : Corrected segmentation masks for the Butterfly dataset from the [Leeds Butterfly Dataset](https://www.josiahwang.com/dataset/leedsbutterfly)

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

### Fuzzy logic
The operators can be computed using different Fuzzy logic operators. It is possible to choose one of the following :
"product", "multi-linear", "minmax", "drastic", "bounded", "einstein", "hamacher"

Product logic is selected if none is specified.

### Filters description
* ***SoftErosion*** : Erode the foreground 
    * <u>forward parameters</u> (*image* = input image, *iterations* = number of times the morphological operation is repeated, *connectivity* = structuring element [4, 8] in 2D and [6, 18, 26] in 3D, *method* = fuzzy logic operator selected to perform the operation)
* ***SoftDilation*** : Dilate the foreground
    * <u>forward parameters</u> (*image* = input image, *iterations* = number of times the morphological operation is repeated, *connectivity* = structuring element [4, 8] in 2D and [6, 18, 26] in 3D, *method* = fuzzy logic operator selected to perform the operation)
* ***SoftClosing*** : Dilation followed by an Erosion
    * <u>forward parameters</u> (*image* = input image, *iterations* = number of times each morphological operation is repeated, *dilation_connectivity* = structuring element [4, 8] in 2D and [6, 18, 26] in 3D for dilation operation, *erosion_connectivity* = structuring element for erosion, *method* = fuzzy logic operator selected to perform the operation)
* ***SoftOpening*** : Erosion followed by a Dilation
    * <u>forward parameters</u> (*image* = input image, *iterations* = number of times each morphological operation is repeated, *dilation_connectivity* = structuring element [4, 8] in 2D and [6, 18, 26] in 3D for dilation operation, *erosion_connectivity* = structuring element for erosion, *method* = fuzzy logic operator selected to perform the operation)
* ***SoftSkeletonizer*** : Repeated thinning operation to extract the centerline of the foreground
    * <u>init parameters</u> (*max_iter* = maximum number of repeated thinning operations, *stop* = maximum percentage of change between two thinning operators compared to initial object to stop the operations.)
    * <u>forward parameters</u> (*image* = input image, *method* = fuzzy logic operator selected to perform the operation)



## Contact
For questions or inquiries, please contact Lisa Guzzi at lisa.guzzi@inria.fr 
