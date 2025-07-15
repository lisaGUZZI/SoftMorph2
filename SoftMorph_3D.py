import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMorphologyBase(nn.Module):
    """Base class for soft morphological operations providing common fuzzy logic methods and utilities."""
    
    def __init__(self):
        super(SoftMorphologyBase, self).__init__()
        self.cube_size = 3  # For 3D operations
    
    def test_format(self, img, connectivity=None, method="product"):
        """Validates and formats input image dimensions and parameters."""
        dim = img.dim()
        size = img.size()
        
        # Check for 3D or 2D based on dimension count
        if dim > 5 or dim < 2:
            raise Exception(f"Invalid input shape {size}. Expected [batch_size, channels, depth, height, width] or [depth, height, width] for 3D, [batch_size, channels, height, width] or [height, width] for 2D.")
        
        # Handle 3D case
        if dim >= 3:
            if dim < 5:
                if dim == 4:
                    if size[0] > 3:
                        raise Exception(f"Ambiguous input shape {size}. Expected [batch_size, channels, depth, height, width] or [depth, height, width].")
                for i in range(5-dim):
                    img = img.unsqueeze(0)
                print("Image resized to:", img.size())
        # Handle 2D case
        else:
            if dim < 4:
                if dim == 3:
                    if size[0] > 3:
                        raise Exception(f"Ambiguous input shape {size}. Expected [batch_size, channels, height, width] or [height, width].")
                for i in range(4-dim):
                    img = img.unsqueeze(0)
                print("Image resized to:", img.size())
        
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Input image values must be in the range [0, 1].")
        
        # Validate connectivity based on dimensionality
        if connectivity is not None:
            if len(img.shape) == 5:  # 3D case
                if connectivity not in [6, 18, 26]:
                    raise ValueError("3D connectivity should be 6, 18, or 26")
            else:  # 2D case
                if connectivity not in [4, 8]:
                    raise ValueError("2D connectivity should be 4 or 8")
        
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


class SoftErosion3D(SoftMorphologyBase):
    """Differentiable soft erosion operation for 3D images."""
    
    def __init__(self):
        super(SoftErosion3D, self).__init__()
        self.indices_list = torch.tensor([
            [2,0,0], [2,0,1], [2,0,2], [1,0,2], [0,0,2], [0,0,1], [0,0,0], [1,0,0], [1,0,1],
            [2,1,0], [2,1,1], [2,1,2], [1,1,2], [0,1,2], [0,1,1], [0,1,0], [1,1,0],
            [2,2,0], [2,2,1], [2,2,2], [1,2,2], [0,2,2], [0,2,1], [0,2,0], [1,2,0], [1,2,1], [1,1,1]
        ], dtype=torch.long)

    def allcondArithm(self, n, connectivity, method):
        """Applies erosion formula to 3x3x3 neighborhoods based on connectivity and method."""
        if connectivity == 6:  
            vox = [8, 10, 12, 25, 16, 14, 26]
        elif connectivity == 18:
            vox = [8, 10, 12, 25, 16, 14, 1,3,5,7,9,11,13,15,18,20,22,24, 26]
        else:
            vox = [8, 10, 12, 25, 16, 14, 1,3,5,7,9,11,13,15,18,20,22,24,0,2,4,6,17,19,21,23,26]

        if method == "product" or method == "multi-linear": 
            F = torch.prod(n[:, :, :, vox], dim=-1)
        else:
            functions = {"minmax": self.minmax, "drastic": self.drastic, "bounded": self.elBoundDiff, "einstein": self.elEinstein, "hamacher": self.elHamacher}
            funct = functions[method]
            neighbor_list = [n[:, :, :, i] for i in vox]
            F = funct(neighbor_list, ope=0)
        return F

    def forward(self, im, iterations=1, connectivity=6, method="product"):
        """Performs soft erosion on input 3D image for specified iterations."""
        im = self.test_format(im, connectivity, method)
        for _ in range(iterations):
            unfolded = torch.nn.functional.pad(im, (1, 1, 1, 1, 1, 1), mode='constant', value=1)
            unfolded = unfolded.unfold(2, self.cube_size, 1).unfold(3, self.cube_size, 1).unfold(4, self.cube_size, 1)
            unfolded = unfolded.contiguous().view(im.shape[0], im.shape[1], (im.shape[2]*im.shape[3]*im.shape[4]), (self.cube_size**3)) 
            unfolded = unfolded[:, :, :, (self.indices_list[:, 0] * 9) + (self.indices_list[:, 1] * 3) + self.indices_list[:, 2]]
            output = self.allcondArithm(unfolded, connectivity, method)
            output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3], im.shape[4])
            im = im * output
        return im


class SoftDilation3D(SoftMorphologyBase):
    """Differentiable soft dilation operation for 3D images."""
    
    def __init__(self):
        super(SoftDilation3D, self).__init__()
        self.indices_list = torch.tensor([
            [2,0,0], [2,0,1], [2,0,2], [1,0,2], [0,0,2], [0,0,1], [0,0,0], [1,0,0], [1,0,1],
            [2,1,0], [2,1,1], [2,1,2], [1,1,2], [0,1,2], [0,1,1], [0,1,0], [1,1,0],
            [2,2,0], [2,2,1], [2,2,2], [1,2,2], [0,2,2], [0,2,1], [0,2,0], [1,2,0], [1,2,1], [1,1,1]
        ], dtype=torch.long)

    def allcondArithm(self, n, connectivity, method):
        """Applies dilation formula to 3x3x3 neighborhoods based on connectivity and method."""
        if connectivity == 6:  
            vox = [8, 10, 12, 25, 16, 14, 26]
        elif connectivity == 18:
            vox = [8, 10, 12, 25, 16, 14, 1,3,5,7,9,11,13,15,18,20,22,24, 26]
        else:
            vox = [8, 10, 12, 25, 16, 14, 1,3,5,7,9,11,13,15,18,20,22,24,0,2,4,6,17,19,21,23, 26]
        
        if method == "product" or method == "multi-linear":
            F = 1 - torch.prod(1 - n[:, :, :, vox], dim=-1)
        else:
            functions = {"minmax": self.minmax, "drastic": self.drastic, "bounded": self.elBoundDiff, "einstein": self.elEinstein, "hamacher": self.elHamacher}
            funct = functions[method]
            neighbor_list = [n[:, :, :, i] for i in vox]
            F = funct(neighbor_list, ope=1)
        return F

    def forward(self, im, iterations=1, connectivity=6, method="product"):
        """Performs soft dilation on input 3D image for specified iterations."""
        im = self.test_format(im, connectivity, method)
        for _ in range(iterations):
            unfolded = torch.nn.functional.pad(im, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
            unfolded = unfolded.unfold(2, self.cube_size, 1).unfold(3, self.cube_size, 1).unfold(4, self.cube_size, 1)
            unfolded = unfolded.contiguous().view(im.shape[0], im.shape[1], (im.shape[2]*im.shape[3]*im.shape[4]), (self.cube_size**3)) 
            unfolded = unfolded[:, :, :, (self.indices_list[:, 0] * 9) + (self.indices_list[:, 1] * 3) + self.indices_list[:, 2]]
            output = self.allcondArithm(unfolded, connectivity, method)
            im = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3], im.shape[4])
        return im


class SoftClosing3D(SoftMorphologyBase):
    """Differentiable soft closing operation (dilation followed by erosion) for 3D images."""
    
    def __init__(self):
        super(SoftClosing3D, self).__init__()
        self.dilate = SoftDilation3D()
        self.erode = SoftErosion3D()

    def forward(self, input_img, iterations, dilation_connectivity=6, erosion_connectivity=6, method="product"):
        """Performs soft closing by applying dilation then erosion."""
        output = self.dilate(input_img, iterations, dilation_connectivity, method)
        output = self.erode(output, iterations, erosion_connectivity, method)
        return output


class SoftOpening3D(SoftMorphologyBase):
    """Differentiable soft opening operation (erosion followed by dilation) for 3D images."""
    
    def __init__(self):
        super(SoftOpening3D, self).__init__()
        self.erode = SoftErosion3D()
        self.dilate = SoftDilation3D()

    def forward(self, input_img, iterations, dilation_connectivity=6, erosion_connectivity=6, method="product"):
        """Performs soft opening by applying erosion then dilation."""
        output = self.erode(input_img, iterations, erosion_connectivity, method)
        output = self.dilate(output, iterations, dilation_connectivity, method)
        return output


class SoftSkeletonizer3D(SoftMorphologyBase):
    """Differentiable soft skeletonization operation for 3D images using iterative thinning."""
    
    def __init__(self, max_iter=5):
        super(SoftSkeletonizer3D, self).__init__()
        self.maxiter = max_iter
        self.indices_list = [self.extract_indices(o) for o in range(6)]
        
    def extract_indices(self, o):
        """Extracts ordered indices for each 3D orientation (Up, East, South, Down, West, North)."""
        ind = [
            # Up
            torch.tensor([
                [2,0,0], [2,0,1], [2,0,2], [1,0,2], [0,0,2], [0,0,1], [0,0,0], [1,0,0], [1,0,1],
                [2,1,0], [2,1,1], [2,1,2], [1,1,2], [0,1,2], [0,1,1], [0,1,0], [1,1,0],
                [2,2,0], [2,2,1], [2,2,2], [1,2,2], [0,2,2], [0,2,1], [0,2,0], [1,2,0], [1,2,1]
            ], dtype=torch.long),
            # East
            torch.tensor([
                [2,0,2], [2,1,2], [2,2,2], [1,2,2], [0,2,2], [0,1,2], [0,0,2], [1,0,2], [1,1,2],
                [2,0,1], [2,1,1], [2,2,1], [1,2,1], [0,2,1], [0,1,1], [0,0,1], [1,0,1],
                [2,0,0], [2,1,0], [2,2,0], [1,2,0], [0,2,0], [0,1,0], [0,0,0], [1,0,0], [1,1,0]
            ], dtype=torch.long),
            # South
            torch.tensor([
                [0,0,0], [0,0,1], [0,0,2], [0,1,2], [0,2,2], [0,2,1], [0,2,0], [0,1,0], [0,1,1],
                [1,0,0], [1,0,1], [1,0,2], [1,1,2], [1,2,2], [1,2,1], [1,2,0], [1,1,0],
                [2,0,0], [2,0,1], [2,0,2], [2,1,2], [2,2,2], [2,2,1], [2,2,0], [2,1,0], [2,1,1]
            ], dtype=torch.long),
            # Down
            torch.tensor([
                [0,2,0], [0,2,1], [0,2,2], [1,2,2], [2,2,2], [2,2,1], [2,2,0], [1,2,0], [1,2,1],
                [0,1,0], [0,1,1], [0,1,2], [1,1,2], [2,1,2], [2,1,1], [2,1,0], [1,1,0],
                [0,0,0], [0,0,1], [0,0,2], [1,0,2], [2,0,2], [2,0,1], [2,0,0], [1,0,0], [1,0,1]
            ], dtype=torch.long),
            # West
            torch.tensor([
                [2,2,0], [2,1,0], [2,0,0], [1,0,0], [0,0,0], [0,1,0], [0,2,0], [1,2,0], [1,1,0],
                [2,2,1], [2,1,1], [2,0,1], [1,0,1], [0,0,1], [0,1,1], [0,2,1], [1,2,1],
                [2,2,2], [2,1,2], [2,0,2], [1,0,2], [0,0,2], [0,1,2], [0,2,2], [1,2,2], [1,1,2]
            ], dtype=torch.long),
            # North
            torch.tensor([
                [2,2,0], [2,2,1], [2,2,2], [2,1,2], [2,0,2], [2,0,1], [2,0,0], [2,1,0], [2,1,1],
                [1,2,0], [1,2,1], [1,2,2], [1,1,2], [1,0,2], [1,0,1], [1,0,0], [1,1,0],
                [0,2,0], [0,2,1], [0,2,2], [0,1,2], [0,0,2], [0,0,1], [0,0,0], [0,1,0], [0,1,1]
            ], dtype=torch.long) 
        ]
        return ind[o]

    def allcondArithm(self, n, method):
        """Applies 3D thinning formula to 3x3x3 neighborhoods using specified fuzzy logic method."""
        if method == "product":
            M1 = (1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*(1-n[:, :, :, 4])*(1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*(1-n[:, :, :, 8])*n[:, :, :, 25]*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 10])*(1-n[:, :, :, 11])*(1-n[:, :, :, 12])*(1-n[:, :, :, 13])*(1-n[:, :, :, 14])*(1-n[:, :, :, 15])*(1-n[:, :, :, 16])*(1-n[:, :, :, 17])*(1-n[:, :, :, 18])*(1-n[:, :, :, 19])*(1-n[:, :, :, 20])*(1-n[:, :, :, 21])*(1-n[:, :, :, 22])*(1-n[:, :, :, 23])*(1-n[:, :, :, 24])))
            M2 = ((1-n[:, :, :, 8])*n[:, :, :, 25]) * (1-((1-((1-n[:, :, :, 3])*(1-n[:, :, :, 4])*(1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*n[:, :, :, 10]))*(1-((1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*(1-n[:, :, :, 4])*(1-n[:, :, :, 5])*n[:, :, :, 16]))*(1-((1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*(1-n[:, :, :, 7])*n[:, :, :, 14]))*(1-((1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*n[:, :, :, 12]))))
            M3 = ((1-n[:, :, :, 8])*n[:, :, :, 25]) * (1 - ((1-((1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*n[:, :, :, 10]*n[:, :, :, 12]))*(1-((1-n[:, :, :, 5])*(1-n[:, :, :, 4])*(1-n[:, :, :, 3])*n[:, :, :, 10]*n[:, :, :, 16]))*(1-((1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*n[:, :, :, 16]*n[:, :, :, 14]))*(1-((1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 7])*n[:, :, :, 14]*n[:, :, :, 12]))))
            M4 = ((1-n[:, :, :, 1])*(1-n[:, :, :, 3])*(1-n[:, :, :, 5])*(1-n[:, :, :, 7])*(1-n[:, :, :, 8])*n[:, :, :, 25]) * (1-((1-((1-n[:, :, :, 0])*(1-n[:, :, :, 4])*(1-n[:, :, :, 6])*n[:, :, :, 11]*n[:, :, :, 2]))*(1-((1-n[:, :, :, 2])*(1-n[:, :, :, 4])*(1-n[:, :, :, 6])*n[:, :, :, 0]*n[:, :, :, 9]))*(1-((1-n[:, :, :, 0])*(1-n[:, :, :, 4])*(1-n[:, :, :, 2])*n[:, :, :, 6]*n[:, :, :, 15]))*(1-((1-n[:, :, :, 0])*(1-n[:, :, :, 6])*(1-n[:, :, :, 2])*n[:, :, :, 4]*n[:, :, :, 13]))))
            M5 = ((1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*(1-n[:, :, :, 4])*(1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*(1-n[:, :, :, 8])*(1-n[:, :, :, 25])) * (1 - ((1-((1-n[:, :, :, 13])*(1-n[:, :, :, 14])*(1-n[:, :, :, 15])*(1-n[:, :, :, 21])*(1-n[:, :, :, 22])*(1-n[:, :, :, 23])*n[:, :, :, 18]*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 10])*(1-n[:, :, :, 11])*(1-n[:, :, :, 12])*(1-n[:, :, :, 16])*(1-n[:, :, :, 17])*(1-n[:, :, :, 19])*(1-n[:, :, :, 20])*(1-n[:, :, :, 24])))))*(1-((1-n[:, :, :, 13])*(1-n[:, :, :, 12])*(1-n[:, :, :, 11])*(1-n[:, :, :, 21])*(1-n[:, :, :, 20])*(1-n[:, :, :, 19])*n[:, :, :, 24]*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 10])*(1-n[:, :, :, 14])*(1-n[:, :, :, 15])*(1-n[:, :, :, 16])*(1-n[:, :, :, 17])*(1-n[:, :, :, 18])*(1-n[:, :, :, 22])*(1-n[:, :, :, 23])))))*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 10])*(1-n[:, :, :, 11])*(1-n[:, :, :, 17])*(1-n[:, :, :, 18])*(1-n[:, :, :, 19])*n[:, :, :, 22]*(1-((1-n[:, :, :, 15])*(1-n[:, :, :, 14])*(1-n[:, :, :, 13])*(1-n[:, :, :, 12])*(1-n[:, :, :, 16])*(1-n[:, :, :, 23])*(1-n[:, :, :, 21])*(1-n[:, :, :, 20])*(1-n[:, :, :, 24])))))*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 16])*(1-n[:, :, :, 15])*(1-n[:, :, :, 24])*(1-n[:, :, :, 17])*(1-n[:, :, :, 23])*n[:, :, :, 20]*(1-((1-n[:, :, :, 14])*(1-n[:, :, :, 10])*(1-n[:, :, :, 11])*(1-n[:, :, :, 12])*(1-n[:, :, :, 13])*(1-n[:, :, :, 18])*(1-n[:, :, :, 19])*(1-n[:, :, :, 22])*(1-n[:, :, :, 21])))))))
            M6 = ((1-n[:, :, :, 0])*(1-n[:, :, :, 1])*(1-n[:, :, :, 2])*(1-n[:, :, :, 3])*(1-n[:, :, :, 4])*(1-n[:, :, :, 5])*(1-n[:, :, :, 6])*(1-n[:, :, :, 7])*(1-n[:, :, :, 8])*(1-n[:, :, :, 25])) * (1 - ((1-((1-n[:, :, :, 14])*(1-n[:, :, :, 15])*(1-n[:, :, :, 16])*(1-n[:, :, :, 22])*(1-n[:, :, :, 23])*(1-n[:, :, :, 24])*n[:, :, :, 18]*n[:, :, :, 20]))*(1-((1-n[:, :, :, 14])*(1-n[:, :, :, 12])*(1-n[:, :, :, 13])*(1-n[:, :, :, 22])*(1-n[:, :, :, 20])*(1-n[:, :, :, 21])*n[:, :, :, 18]*n[:, :, :, 24]))*(1-((1-n[:, :, :, 10])*(1-n[:, :, :, 11])*(1-n[:, :, :, 12])*(1-n[:, :, :, 18])*(1-n[:, :, :, 19])*(1-n[:, :, :, 20])*n[:, :, :, 24]*n[:, :, :, 22]))*(1-((1-n[:, :, :, 9])*(1-n[:, :, :, 10])*(1-n[:, :, :, 16])*(1-n[:, :, :, 24])*(1-n[:, :, :, 17])*(1-n[:, :, :, 18])*n[:, :, :, 22]*n[:, :, :, 20]))))
            F = 1-((1-M1)*(1-M2)*(1-M3)*(1-M4)*(1-M5)*(1-M6))
            F = 1-F
        else:
            functions = {"minmax": self.minmax, "drastic": self.drastic, "bounded": self.elBoundDiff, "einstein": self.elEinstein, "hamacher": self.elHamacher}
            funct = functions[method]
            
            # M1: AND of negated values + n[25] + negated OR of values
            M1_and = funct([(1-n[:, :, :, i]) for i in range(9)] + [n[:, :, :, 25]], ope=0)
            M1_or = funct([n[:, :, :, i] for i in range(9, 25)], ope=1)
            M1 = funct([M1_and, (1-M1_or)], ope=0)
            
            # M2: AND of two negated values + negated OR of four terms
            M2_and = funct([(1-n[:, :, :, 8]), n[:, :, :, 25]], ope=0)
            
            # Four terms in M2
            M2_term1 = funct([(1-n[:, :, :, i]) for i in [3,4,5,6,7]] + [n[:, :, :, 10]], ope=0)
            M2_term2 = funct([(1-n[:, :, :, i]) for i in [1,2,3,4,5]] + [n[:, :, :, 16]], ope=0)
            M2_term3 = funct([(1-n[:, :, :, i]) for i in [0,1,2,3,7]] + [n[:, :, :, 14]], ope=0)
            M2_term4 = funct([(1-n[:, :, :, i]) for i in [0,1,5,6,7]] + [n[:, :, :, 12]], ope=0)
            
            M2_or = funct([M2_term1, M2_term2, M2_term3, M2_term4], ope=1)
            M2 = funct([M2_and, (1-M2_or)], ope=0)
            
            # M3: AND of two values + negated OR of four terms
            M3_and = funct([(1-n[:, :, :, 8]), n[:, :, :, 25]], ope=0)
            
            M3_term1 = funct([(1-n[:, :, :, i]) for i in [5,6,7]] + [n[:, :, :, 10], n[:, :, :, 12]], ope=0)
            M3_term2 = funct([(1-n[:, :, :, i]) for i in [5,4,3]] + [n[:, :, :, 10], n[:, :, :, 16]], ope=0)
            M3_term3 = funct([(1-n[:, :, :, i]) for i in [1,2,3]] + [n[:, :, :, 16], n[:, :, :, 14]], ope=0)
            M3_term4 = funct([(1-n[:, :, :, i]) for i in [0,1,7]] + [n[:, :, :, 14], n[:, :, :, 12]], ope=0)
            
            M3_or = funct([M3_term1, M3_term2, M3_term3, M3_term4], ope=1)
            M3 = funct([M3_and, (1-M3_or)], ope=0)
            
            # M4: AND of six values + negated OR of four terms
            M4_and = funct([(1-n[:, :, :, i]) for i in [1,3,5,7,8]] + [n[:, :, :, 25]], ope=0)
            
            M4_term1 = funct([(1-n[:, :, :, i]) for i in [0,4,6]] + [n[:, :, :, 11], n[:, :, :, 2]], ope=0)
            M4_term2 = funct([(1-n[:, :, :, i]) for i in [2,4,6]] + [n[:, :, :, 0], n[:, :, :, 9]], ope=0)
            M4_term3 = funct([(1-n[:, :, :, i]) for i in [0,4,2]] + [n[:, :, :, 6], n[:, :, :, 15]], ope=0)
            M4_term4 = funct([(1-n[:, :, :, i]) for i in [0,6,2]] + [n[:, :, :, 4], n[:, :, :, 13]], ope=0)
            
            M4_or = funct([M4_term1, M4_term2, M4_term3, M4_term4], ope=1)
            M4 = funct([M4_and, (1-M4_or)], ope=0)
            
            # M5: AND of ten negated values + negated OR of four complex terms
            M5_and = funct([(1-n[:, :, :, i]) for i in [0,1,2,3,4,5,6,7,8,25]], ope=0)
            
            # M5 term 1
            M5_t1_and = funct([(1-n[:, :, :, i]) for i in [13,14,15,21,22,23]] + [n[:, :, :, 18]], ope=0)
            M5_t1_or = funct([n[:, :, :, i] for i in [9,10,11,12,16,17,19,20,24]], ope=1)
            M5_term1 = funct([M5_t1_and, (1-M5_t1_or)], ope=0)
            
            # M5 term 2
            M5_t2_and = funct([(1-n[:, :, :, i]) for i in [13,12,11,21,20,19]] + [n[:, :, :, 24]], ope=0)
            M5_t2_or = funct([n[:, :, :, i] for i in [9,10,14,15,16,17,18,22,23]], ope=1)
            M5_term2 = funct([M5_t2_and, (1-M5_t2_or)], ope=0)
            
            # M5 term 3
            M5_t3_and = funct([(1-n[:, :, :, i]) for i in [9,10,11,17,18,19]] + [n[:, :, :, 22]], ope=0)
            M5_t3_or = funct([n[:, :, :, i] for i in [15,14,13,12,16,23,21,20,24]], ope=1)
            M5_term3 = funct([M5_t3_and, (1-M5_t3_or)], ope=0)
            
            # M5 term 4
            M5_t4_and = funct([(1-n[:, :, :, i]) for i in [9,16,15,24,17,23]] + [n[:, :, :, 20]], ope=0)
            M5_t4_or = funct([n[:, :, :, i] for i in [14,10,11,12,13,18,19,22,21]], ope=1)
            M5_term4 = funct([M5_t4_and, (1-M5_t4_or)], ope=0)
            
            M5_or = funct([M5_term1, M5_term2, M5_term3, M5_term4], ope=1)
            M5 = funct([M5_and, (1-M5_or)], ope=0)
            
            # M6: AND of ten negated values + negated OR of four terms
            M6_and = funct([(1-n[:, :, :, i]) for i in [0,1,2,3,4,5,6,7,8,25]], ope=0)
            
            M6_term1 = funct([(1-n[:, :, :, i]) for i in [14,15,16,22,23,24]] + [n[:, :, :, 18], n[:, :, :, 20]], ope=0)
            M6_term2 = funct([(1-n[:, :, :, i]) for i in [14,12,13,22,20,21]] + [n[:, :, :, 18], n[:, :, :, 24]], ope=0)
            M6_term3 = funct([(1-n[:, :, :, i]) for i in [10,11,12,18,19,20]] + [n[:, :, :, 24], n[:, :, :, 22]], ope=0)
            M6_term4 = funct([(1-n[:, :, :, i]) for i in [9,10,16,24,17,18]] + [n[:, :, :, 22], n[:, :, :, 20]], ope=0)
            
            M6_or = funct([M6_term1, M6_term2, M6_term3, M6_term4], ope=1)
            M6 = funct([M6_and, (1-M6_or)], ope=0)
            
            # Final F: negated AND of six negated terms
            F_and = funct([(1-M1), (1-M2), (1-M3), (1-M4), (1-M5), (1-M6)], ope=0)
            F = 1 - (1 - F_and)
        
        return F

    def forward(self, im, method="product"):
        """Performs 3D soft skeletonization using iterative thinning in six directions."""
        im = self.test_format(im)
        for _ in range(self.maxiter):
            for o in range(6):
                unfolded = torch.nn.functional.pad(im, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
                unfolded = unfolded.unfold(2, self.cube_size, 1).unfold(3, self.cube_size, 1).unfold(4, self.cube_size, 1)
                unfolded = unfolded.contiguous().view(im.shape[0], im.shape[1], (im.shape[2]*im.shape[3]*im.shape[4]), (self.cube_size**3)) 
                unfolded = unfolded[:, :, :, (self.indices_list[o][:, 0] * 9) + (self.indices_list[o][:, 1] * 3) + self.indices_list[o][:, 2]]
                output = self.allcondArithm(unfolded, method)
                output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3], im.shape[4])
                im = im * output
        return im
