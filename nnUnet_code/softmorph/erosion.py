import torch
import torch.nn as nn

class Erosion(nn.Module):
    def __init__(self):
        super(Erosion, self).__init__()
        self.cube_size = 3
        self.indices_list = [self.ext_ind(s) for s in range(1)]
    
    def ext_ind(self, o):
        ind = [torch.tensor([
        [2,0,0], [2,0,1], [2,0,2], [1,0,2], [0,0,2], [0,0,1], [0,0,0], [1,0,0], [1,0,1],
        [2,1,0], [2,1,1], [2,1,2], [1,1,2], [0,1,2], [0,1,1], [0,1,0], [1,1,0],
        [2,2,0], [2,2,1], [2,2,2], [1,2,2], [0,2,2], [0,2,1], [0,2,0], [1,2,0], [1,2,1], [1,1,1]
    ], dtype=torch.long),
    torch.tensor([[0,0,2], [1,0,2], [2,0,2], [2,1,2], [2,2,2], [1,2,2], [0,2,2], [0,1,2], [1,1,2],
        [0,0,1], [1,0,1], [2,0,1], [2,1,1], [2,2,1], [1,2,1], [0,2,1], [0,1,1],
        [0,0,0], [1,0,0], [2,0,0], [2,1,0], [2,2,0], [1,2,0], [0,2,0], [0,1,0], [1,1,0], [1,1,1]
        ], dtype=torch.long),
    torch.tensor([[0,0,0], [0,0,1], [0,0,2], [0,1,2], [0,2,2], [0,2,1], [0,2,0], [0,1,0], [0,1,1],
        [1,0,0], [1,0,1], [1,0,2], [1,1,2], [1,2,2], [1,2,1], [1,2,0], [1,1,0],
        [2,0,0], [2,0,1], [2,0,2], [2,1,2], [2,2,2], [2,2,1], [2,2,0], [2,1,0], [2,1,1], [1,1,1]
        ], dtype=torch.long),
    torch.tensor([[0,2,0], [0,2,1], [0,2,2], [1,2,2], [2,2,2], [2,2,1], [2,2,0], [1,2,0], [1,2,1],
        [0,1,0], [0,1,1], [0,1,2], [1,1,2], [2,1,2], [2,1,1], [2,1,0], [1,1,0],
        [0,0,0], [0,0,1], [0,0,2], [1,0,2], [2,0,2], [2,0,1], [2,0,0], [1,0,0], [1,0,1], [1,1,1]
        ], dtype=torch.long),
    torch.tensor([[2,0,0], [1,0,0], [0,0,0], [0,1,0], [0,2,0], [1,2,0], [2,2,0], [2,1,0], [1,1,0],
        [2,0,1], [1,0,1], [0,0,1], [0,1,1], [0,2,1], [1,2,1], [2,2,1], [2,1,1],
        [2,0,2], [1,0,2], [0,0,2], [0,1,2], [0,2,2], [1,2,2], [2,2,2], [2,1,2], [1,1,2], [1,1,1]
        ], dtype=torch.long),
    torch.tensor([[2,0,2], [2,0,1], [2,0,0], [2,1,0], [2,2,0], [2,2,1], [2,2,2], [2,1,2], [2,1,1],
        [1,0,2], [1,0,1], [1,0,0], [1,1,0], [1,2,0], [1,2,1], [1,2,2], [1,1,2],
        [0,0,2], [0,0,1], [0,0,0], [0,1,0], [0,2,0], [0,2,1], [0,2,2], [0,1,2], [0,1,1], [1,1,1]
        ]  , dtype=torch.long) 
]
        indices = ind[o]
        return indices

    def allcondArithm(self, n, connec):
        if connec == 6:
            vox = [8, 10, 12, 25, 16, 14, 26]
        elif connec == 18:
            vox = [8, 10, 12, 25, 16, 14, 1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 22, 24, 26]
        else:
            vox = [
                8,
                10,
                12,
                25,
                16,
                14,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                18,
                20,
                22,
                24,
                0,
                2,
                4,
                6,
                17,
                19,
                21,
                23,
                26,
            ]

        F = torch.prod(n[:, :, :, vox], dim=-1)
        return F

    def forward(self, im, iter=2, connec = 6, padd = 1):
        for i in range(iter):
            unfolded = torch.nn.functional.pad(im, (1, 1, 1, 1, 1, 1), mode='constant', value=padd)
            unfolded = unfolded.unfold(2, self.cube_size, 1).unfold(3, self.cube_size, 1).unfold(4, self.cube_size, 1)
            unfolded= unfolded.contiguous().view(im.shape[0], im.shape[1], (im.shape[2]*im.shape[3]*im.shape[4]), (self.cube_size**3)) 
            unfolded = unfolded[:, :, :,(self.indices_list[0][:, 0] * 9) + (self.indices_list[0][:, 1] * 3) + self.indices_list[0][:, 2]]
            output = self.allcondArithm(unfolded, connec)
            output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3], im.shape[4])
            im = im * output
        return im 