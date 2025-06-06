import torch.nn as nn
from .dilation import Dilation
from .erosion import Erosion

class Opening(nn.Module):
    def __init__(self):
        super(Opening, self).__init__()
        self.morph_dilate = Dilation()
        self.morph_erode = Erosion()
        
    def forward(self, input_img, iter=1, foreground_connec=6, back_connec=6):
        # Erosion
        output = self.morph_erode(input_img, iter, foreground_connec)
        # Dilation
        output = self.morph_dilate(output, iter, back_connec)
        return output 