import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMorphologyBase(nn.Module):
    """Base class for soft morphological operations providing common fuzzy logic methods and utilities."""
    
    def __init__(self):
        super(SoftMorphologyBase, self).__init__()
    
    def test_format(self, img, connectivity=None, method="product"):
        """Validates and formats input image dimensions and parameters."""
        dim = img.dim()
        size = img.size()
        if dim > 4 or dim < 2:
            raise Exception(f"Invalid input shape {size}. Expected [batch_size, channels, height, width] or [height, width].")
        elif dim < 4:
            if dim == 3:
                if size[0] > 3:
                    raise Exception(f"Ambiguous input shape {size}. Expected [batch_size, channels, height, width] or [height, width].")
            for i in range(4-dim):
                img = img.unsqueeze(0)
            print("Image resized to:", img.size())
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Input image values must be in the range [0, 1].")
        if connectivity is not None and connectivity != 4 and connectivity != 8:
            raise ValueError("Connectivity should either be 4 or 8")
        if method not in ["product", "multi-linear", "minmax", "drastic", "bounded", "einstein", "hamacher"]:
            raise ValueError("Invalid method. Choose among 'product', 'multi-linear', 'minmax', 'drastic', 'bounded', 'einstein', 'hamacher'")
        return img
    
    def drastic(self, elements, ope):
        """Applies drastic fuzzy logic operation (T-norm or S-norm) to list of elements."""
        max_val = torch.max(torch.stack(elements, dim=-1), dim=-1)[0]
        min_val = torch.min(torch.stack(elements, dim=-1), dim=-1)[0]
        if ope == 0:
            return torch.where(max_val == 1, min_val, torch.tensor(0))
        elif ope == 1:
            return torch.where(min_val == 0, max_val, torch.tensor(1))
        
    def minmax(self, elements, ope):
        """Applies min-max fuzzy logic operation to list of elements."""
        if ope == 0:
            return torch.min(torch.stack(elements, dim=-1), dim=-1)[0]
        elif ope == 1:
            return torch.max(torch.stack(elements, dim=-1), dim=-1)[0]
    
    def boundDiff(self, A, B, ope):
        """Applies bounded difference fuzzy logic operation to two elements."""
        a = A + B
        if ope == 0:
            return torch.max(torch.zeros_like(a), (a-1))
        elif ope == 1:
            return torch.min(torch.ones_like(a), a)
        
    def elBoundDiff(self, elements, ope):
        """Applies bounded difference operation recursively to list of elements."""
        tot = self.boundDiff(elements[0], elements[1], ope)
        if len(elements) > 2:
            for i in range(2, len(elements)):
                tot = self.boundDiff(tot, elements[i], ope)
        return tot
    
    def einstein(self, A, B, ope):
        """Applies Einstein fuzzy logic operation to two elements."""
        if ope == 0:
            return (A*B)/(2-(A+B-(A*B)))
        elif ope == 1:
            return (A+B)/(1+(A*B))
        
    def elEinstein(self, elements, ope):
        """Applies Einstein operation recursively to list of elements."""
        tot = self.einstein(elements[0], elements[1], ope)
        if len(elements) > 2:
            for i in range(2, len(elements)):
                tot = self.einstein(tot, elements[i], ope)
        return tot
    
    def hamacher(self, A, B, ope):
        """Applies Hamacher fuzzy logic operation to two elements."""
        epsilon = 1e-8
        if ope == 0:
            ab = A * B
            denominator = A + B - ab
            result = torch.where(
                (torch.abs(A) < epsilon) & (torch.abs(B) < epsilon),
                torch.zeros_like(A),
                ab / (denominator + epsilon)
            )
            return result
        elif ope == 1:
            ab = A * B
            numerator = A + B - (2 * ab)
            denominator = 1 - ab
            result = torch.where(
                (torch.abs(1-A) < epsilon) & (torch.abs(1-B) < epsilon),
                torch.ones_like(A),
                numerator / (denominator + epsilon)
            )
            return result
        
    def elHamacher(self, elements, ope):
        """Applies Hamacher operation recursively to list of elements."""
        tot = self.hamacher(elements[0], elements[1], ope)
        if len(elements) > 2:
            for i in range(2, len(elements)):
                tot = self.hamacher(tot, elements[i], ope)
        return tot


class SoftDilation(SoftMorphologyBase):
    """Differentiable soft dilation operation for 2D images."""
    
    def __init__(self):
        super(SoftDilation, self).__init__()
        self.indices_list = [self.ext_ind(s) for s in range(1)]
    
    def ext_ind(self, o):
        """Extracts indices for 8-connected neighborhood in specified orientation."""
        indices = torch.tensor([
            [0, 1], [0, 2], [1, 2], [2, 2],
            [2, 1], [2, 0], [1, 0], [0, 0], [1, 1]
        ], dtype=torch.long)
        indices = torch.roll(indices, -2 * o, dims=0)
        return indices

    def allcondArithm(self, n, connec, method):
        """Applies dilation formula to 3x3 neighborhoods based on connectivity and method."""
        if method == "product" or method == "multi-linear":   
            if connec == 4:
                F = 1 - torch.prod(1 - n[:, :, :, ::2], dim=-1)
            else:
                F = 1 - torch.prod(1 - n[:, :, :, :], dim=-1)
        else: 
            functions = {"minmax": self.minmax, "drastic": self.drastic, "bounded": self.elBoundDiff, "einstein": self.elEinstein, "hamacher": self.elHamacher}
            funct = functions[method]
            if connec == 4:
                neighbor_list = [n[:, :, :, i] for i in range(0, n.shape[-1], 2)]
                F = funct(neighbor_list, ope=1)
            else:
                neighbor_list = [n[:, :, :, i] for i in range(n.shape[-1])]
                F = funct(neighbor_list, ope=1)
        return F

    def forward(self, im, iterations=1, connectivity=4, method="product"):
        """Performs soft dilation on input image for specified iterations."""
        for _ in range(iterations):
            unf = nn.Unfold((im.shape[2], im.shape[3]), 1, 1, 1)
            unfolded = unf(im) 
            unfolded = unfolded.view(im.shape[0], im.shape[1], -1, unfolded.size(-1))
            unfolded = unfolded[:, :, :, (self.indices_list[0][:, 0] * 3) + self.indices_list[0][:, 1]]
            output = self.allcondArithm(unfolded, connectivity, method)
            output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3])
            im = output
        return im


class SoftErosion(SoftMorphologyBase):
    """Differentiable soft erosion operation for 2D images."""
    
    def __init__(self):
        super(SoftErosion, self).__init__()
        self.indices_list = torch.tensor([          
            [0, 1], [0, 2], [1, 2], [2, 2],
            [2, 1], [2, 0], [1, 0], [0, 0], [1, 1]
        ], dtype=torch.long)

    def allcondArithm(self, n, connectivity, method):
        """Applies erosion formula to 3x3 neighborhoods based on connectivity and method."""
        if method == "product" or method == "multi-linear": 
            if connectivity == 4:  
                F = torch.prod(n[:, :, :, ::2], dim=-1)
            else: 
                F = torch.prod(n, dim=-1)
        else:
            functions = {"minmax": self.minmax, "drastic": self.drastic, "bounded": self.elBoundDiff, "einstein": self.elEinstein, "hamacher": self.elHamacher}
            funct = functions[method]
            if connectivity == 4:
                neighbor_list = [n[:, :, :, i] for i in range(0, n.shape[-1], 2)]
                F = funct(neighbor_list, ope=0)
            else:
                neighbor_list = [n[:, :, :, i] for i in range(n.shape[-1])]
                F = funct(neighbor_list, ope=0)
        return F

    def forward(self, im, iterations=1, connectivity=4, method="product"):
        """Performs soft erosion on input image for specified iterations."""
        for _ in range(iterations):
            im_padded = F.pad(im, (1, 1, 1, 1), mode='constant', value=1)
            unf = nn.Unfold((im.shape[2], im.shape[3]), 1, 0, 1)
            unfolded = unf(im_padded) 
            unfolded = unfolded.view(im.shape[0], im.shape[1], -1, unfolded.size(-1))
            unfolded = unfolded[:, :, :, (self.indices_list[:, 0] * 3) + self.indices_list[:, 1]]
            output = self.allcondArithm(unfolded, connectivity, method)
            output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3])
            im = im * output
        return im


class SoftClosing(SoftMorphologyBase):
    """Differentiable soft closing operation (dilation followed by erosion) for 2D images."""
    
    def __init__(self):
        super(SoftClosing, self).__init__()
        self.dilate = SoftDilation()
        self.erode = SoftErosion()

    def forward(self, input_img, iterations, dilation_connectivity=4, erosion_connectivity=4, method="product"):
        """Performs soft closing by applying dilation then erosion."""
        output = self.dilate(input_img, iterations, dilation_connectivity, method)
        output = self.erode(output, iterations, erosion_connectivity, method)
        return output


class SoftOpening(SoftMorphologyBase):
    """Differentiable soft opening operation (erosion followed by dilation) for 2D images."""
    
    def __init__(self):
        super(SoftOpening, self).__init__()
        self.erode = SoftErosion()
        self.dilate = SoftDilation()

    def forward(self, input_img, iterations, dilation_connectivity=4, erosion_connectivity=4, method="product"):
        """Performs soft opening by applying erosion then dilation."""
        output = self.erode(input_img, iterations, erosion_connectivity, method)
        output = self.dilate(output, iterations, dilation_connectivity, method)
        return output


class SoftSkeletonizer(SoftMorphologyBase):
    """Differentiable soft skeletonization operation for 2D images using iterative thinning."""
    
    def __init__(self, max_iter=100, stop=0.02):
        super(SoftSkeletonizer, self).__init__()
        self.maxiter = max_iter
        self.stop = stop
        self.indices_list = [self.extract_indices(o) for o in range(4)]
        
    def extract_indices(self, o):
        """Extracts ordered indices for each orientation (North, East, South, West)."""
        indices = torch.tensor([
            [0, 1], [0, 2], [1, 2], [2, 2],
            [2, 1], [2, 0], [1, 0], [0, 0]
        ], dtype=torch.long)
        indices = torch.roll(indices, -2 * o, dims=0)
        return indices

    def allcondArithm(self, n, method):
        """Applies thinning formula to 3x3 neighborhoods using specified fuzzy logic method."""
        if method == "product":
            F1 = (1 - n[:, :, :, 0])
            F2 = ((1 - n[:, :, :, 1]) * (1 - n[:, :, :, 7]) * 
                  (1-n[:, :, :, 2]-n[:, :, :, 3]+2*n[:, :, :, 2]*n[:, :, :, 3]-n[:, :, :, 4]+2*n[:, :, :, 2]*n[:, :, :, 4]+2*n[:, :, :, 3]*n[:, :, :, 4]-4*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]) *
                (n[:, :, :, 3] + n[:, :, :, 5] - 2 * (n[:, :, :, 3] * n[:, :, :, 5])) * (n[:, :, :, 3] + n[:, :, :, 6] - 2 * (n[:, :, :, 3] * n[:, :, :, 6])))
            F3 = ((n[:, :, :, 1] + n[:, :, :, 5] - 2 * (n[:, :, :, 1] * n[:, :, :, 5])) * (n[:, :, :, 2] + n[:, :, :, 5] - 2 * (n[:, :, :, 2] * n[:, :, :, 5])) *
                (n[:, :, :, 4] + (1 - n[:, :, :, 5]) - 2 * (n[:, :, :, 4] * (1 - n[:, :, :, 5]))) * (1 - n[:, :, :, 6]) * (1 - n[:, :, :, 7]))
            F4 = (n[:, :, :, 2] * n[:, :, :, 4] * (1 - n[:, :, :, 7]))
            F5 = ((1 - n[:, :, :, 1]) * n[:, :, :, 4] * n[:, :, :, 6])
            F6 = ((1 - n[:, :, :, 1]) * (1 - n[:, :, :, 2]) * (1 - n[:, :, :, 3]) * n[:, :, :, 6] * n[:, :, :, 7])
            F = 1-(F1 * (1 - ((1 - F2) * (1 - F3) * (1 - F4) * (1 - F5) * (1 - F6) * (1 - F6))))
        elif method == "multi-linear":
            F = (n[:, :, :, 0] - 1)*(3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5] - 3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 7] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 6]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3] - 3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + 3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5] + 3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 5]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 5] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 6] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 2] - 2*n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5] + 2*n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4] + n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 5] - n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 1]*n[:, :, :, 6]*n[:, :, :, 7] - 3*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + 3*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] + 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] - 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5] + 3*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6] - 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 7] + 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4] + 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 7] + n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5] - 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 6] + n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 7] - n[:, :, :, 2]*n[:, :, :, 3] + 2*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] + n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5] - 2*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 6] + n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 7] - n[:, :, :, 2]*n[:, :, :, 4] - n[:, :, :, 2]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 2]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 2]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] + n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5] - 2*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6] + n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 7] - n[:, :, :, 3]*n[:, :, :, 4] - n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 3]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] - n[:, :, :, 4]*n[:, :, :, 5] + n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] - n[:, :, :, 4]*n[:, :, :, 6] + n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 6]*n[:, :, :, 7])
            F = 1-F
        else:
            functions = {"minmax": self.minmax, "drastic": self.drastic, "bounded": self.elBoundDiff, "einstein": self.elEinstein, "hamacher": self.elHamacher}
            funct = functions[method]
            
            F1 = (1 - n[:, :, :, 0])
            F2 = funct([
                (1 - n[:, :, :, 1]), (1 - n[:, :, :, 7]),
                funct([
                    funct([n[:, :, :, 2], (1-n[:, :, :, 3]), n[:, :, :, 4]], ope=0),
                    funct([(1-n[:, :, :, 2]), n[:, :, :, 3], n[:, :, :, 4]], ope=0),
                    funct([(1-n[:, :, :, 2]), (1-n[:, :, :, 3]), (1-n[:, :, :, 4])], ope=0),
                    funct([n[:, :, :, 2], n[:, :, :, 3], (1-n[:, :, :, 4])], ope=0)
                ], ope=1),
                funct([
                    funct([n[:, :, :, 3], (1 - n[:, :, :, 5])], ope=0),
                    funct([(1 - n[:, :, :, 3]), n[:, :, :, 5]], ope=0)
                ], ope=1),
                funct([
                    funct([n[:, :, :, 3], (1 - n[:, :, :, 6])], ope=0),
                    funct([(1 - n[:, :, :, 3]), n[:, :, :, 6]], ope=0)
                ], ope=1)
            ], ope=0)
            F3 = funct([
                funct([
                    funct([n[:, :, :, 1], (1-n[:, :, :, 5])], ope=0),
                    funct([(1-n[:, :, :, 1]), n[:, :, :, 5]], ope=0)
                ], ope=1),
                funct([
                    funct([n[:, :, :, 2], (1-n[:, :, :, 5])], ope=0),
                    funct([(1 - n[:, :, :, 2]), n[:, :, :, 5]], ope=0)
                ], ope=1),
                funct([
                    funct([n[:, :, :, 4], n[:, :, :, 5]], ope=0),
                    funct([(1 - n[:, :, :, 4]), (1-n[:, :, :, 5])], ope=0)
                ], ope=1),
                (1 - n[:, :, :, 6]), (1 - n[:, :, :, 7])
            ], ope=0)
            F4 = funct([n[:, :, :, 2], n[:, :, :, 4], (1 - n[:, :, :, 7])], ope=0)
            F5 = funct([(1 - n[:, :, :, 1]), n[:, :, :, 4], n[:, :, :, 6]], ope=0)
            F6 = funct([(1 - n[:, :, :, 1]), (1 - n[:, :, :, 2]), (1 - n[:, :, :, 3]), n[:, :, :, 6], n[:, :, :, 7]], ope=0)
            F = 1 - funct([F1, funct([F2, F3, F4, F5, F6], ope=1)], ope=0)
        return F
    
    def testchange(self, s1, s2, obj):
        """Tests if pixel change between iterations is below threshold."""
        rest = abs(s1-s2)
        final = (torch.sum(rest))/obj
        if final <= self.stop:
            self.change = False

    def forward(self, im, method="product"):
        """Performs soft skeletonization using iterative thinning in four directions."""
        im = self.test_format(im, method=method)
        obj = torch.sum(im)
        self.change = True
        count = 0
        while self.change and count < self.maxiter:
            count += 1
            image = im.clone()
            for o in range(4):
                unf = nn.Unfold((im.shape[2], im.shape[3]), 1, 1, 1)
                unfolded = unf(im) 
                unfolded = unfolded.view(im.shape[0], im.shape[1], -1, unfolded.size(-1))
                unfolded = unfolded[:, :, :, (self.indices_list[o][:, 0] * 3) + self.indices_list[o][:, 1]]
                output = self.allcondArithm(unfolded, method)
                output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3])
                im = im * output
            self.testchange(image, im, obj)
        return im
