import torch.nn as nn
from .dilation import Dilation
from .erosion import Erosion

class Closing(nn.Module):
    def __init__(self):
        super(Closing, self).__init__()
        self.morph_dilate = Dilation()
        self.morph_erode = Erosion()
        
    def forward(self, input_img, iter=2, foreground_connec=6, back_connec=6):
        # Dilation
        output = self.morph_dilate(input_img, iter, foreground_connec)
        # Erosion
        output = self.morph_erode(output, iter, back_connec, padd=1)
        return output 