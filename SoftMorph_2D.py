import torch
import torch.nn as nn
import torch.nn.functional as F



# Fuzzy logic operators
def drastic(self, elements, ope):
    """elements : liste of elements to compare, 
    ope : type of operation. 0 → product, 1 → sum"""
    
    max_val = torch.max(torch.stack(elements, dim = -1), dim=-1)[0]  # Compute the maximum value along the last dimension
    min_val = torch.min(torch.stack(elements, dim = -1), dim=-1)[0] # Compute the minimum value along the last dimension
    if ope == 0:
        return torch.where(max_val == 1, min_val, torch.tensor(0))
    elif ope == 1:
        return torch.where(min_val == 0, max_val, torch.tensor(1))
        
def minmax(self, elements, ope):
    """elements : liste of elements to compare, 
    ope : type of operation. 0 → product, 1 → sum"""
    if ope == 0:
        return torch.min(torch.stack(elements, dim = -1), dim=-1)[0]
    elif ope ==1 :
        return torch.max(torch.stack(elements, dim = -1), dim=-1)[0]

def boundDiff(self, A, B, ope):
    """A, B: elements to compare, 
    ope : type of operation. 0 → product, 1 → sum"""
    a = A+B
    if ope == 0:
        return torch.max(torch.zeros_like(a), (a-1))
    elif ope == 1:
        return torch.min(torch.ones_like(a), a)
    
def elBoundDiff(self, elements, ope):
    tot = self.boundDiff(elements[0], elements[1], ope) 
    if len(elements)>2:
        for i in range(2,len(elements)):
            tot = self.boundDiff(tot, elements[i], ope)
    return tot
        

def einstein(self, A, B, ope):
    """A, B: elements to compare, 
    ope : type of operation. 0 → product, 1 → sum"""
    if ope == 0:
        return (A*B)/(2-(A+B-(A*B)))
    elif ope == 1:
        return (A+B)/(1+(A*B))
    
def elEinstein(self, elements, ope):
    tot = self.einstein(elements[0], elements[1], ope) 
    if len(elements)>2:
        for i in range(2,len(elements)):
            tot = self.einstein(tot, elements[i], ope)
    return tot

def hamacher(self, A, B, ope):
    """A, B: elements to compare, 
    ope : type of operation. 0 → product, 1 → sum"""
    if ope == 0:
        result = torch.zeros_like(A)  # Initialize result tensor
        result[((A != 0) | (B != 0))] = (A[((A != 0) | (B != 0))]*B[((A != 0) | (B != 0))]) / ((A[((A != 0) | (B != 0))]+B[((A != 0) | (B != 0))]-(A[((A != 0) | (B != 0))]*B[((A != 0) | (B != 0))]))+1e-10)
        return result
    elif ope == 1:
        result = torch.ones_like(A)  # Initialize result tensor
        result[(A != 1) | (B != 1)] = (A[(A != 1) | (B != 1)]+B[(A != 1) | (B != 1)]-(2*A[(A != 1) | (B != 1)]*B[(A != 1) | (B != 1)])) / ((1-(A[(A != 1) | (B != 1)]*B[(A != 1) | (B != 1)]))+1e-10)
        return result
    
def elHamacher(self, elements, ope):
    tot = self.hamacher(elements[0], elements[1], ope) 
    if len(elements)>2:
        for i in range(2,len(elements)):
            tot = self.hamacher(tot, elements[i], ope)
    return tot

class SoftErosion(nn.Module):
    """
    Class implemented using Pytorch module to perform differentiable soft erosion on 2D input image.
    """
    def __init__(self):
        super(SoftErosion, self).__init__()
        self.indices_list = torch.tensor([          
            [0, 1], [0, 2], [1, 2], [2, 2],
            [2, 1], [2, 0], [1, 0], [0, 0], [1, 1]
        ], dtype=torch.long)
        
    def test_format(self, img, connectivity, method):
        """
        Function to check user inputs :
        - Input image shape must either be [batch_size, channels, height, width] or [height, width]. 
        - Input image values must be between 0 and 1.
        - Connectivity represents the sutructuring element of the operation. In 2D, it must be either 4 or 8
        - Method represents the fuzzy logic method used to perform the operation.
        """
        dim = img.dim()
        size = img.size()
        if dim > 4 or dim <2:
            raise Exception(f"Invalid input shape {size}. Expected [batch_size, channels, height, width] or [height, width]. Consider using the 3D version for 3D input images")
        elif dim < 4:
            if dim ==3 :
                # If the input dimension is 3 it might be due to input format [channels, height width]
                if size[0] > 3 : # If this is not likely we raise an exception.
                    raise Exception(f"Ambiguous input shape {size}. Expected [batch_size, channels, height, width] or [height, width].")
            for i in range(4-dim):img = img.unsqueeze(0) 
            print("Image resized to : ", img.size())
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Input image values must be in the range [0, 1].")
        if connectivity != 4 and connectivity !=8:
            raise ValueError("Connectivity should either be 4 or 8")
        if method not in ["product", "multi-linear", "minmax", "drastic", "bounded", "einstein", "hamacher"]:
            raise ValueError("Unvalid thinning method. Choose among 'product', 'multi-linear', 'minmax', 'drastic', 'bounded', 'einstein', 'hamacher'")
        return img

    def allcondArithm(self, n, connectivity, method):
        """
        Apply polynomial formula based on the boolean expression that defines an erosion on each 3x3 overlapping squares of the 2D image.
        Inputs : vector of 3x3 overlapping squares n, connectivity (4 or 8) defining the structuring element.
        Output : In binary case returns 0 if the central pixel needs to be changed to 0, returns 1 otherwise.
        """ 
        if method == "product" or method == "multi-linear": 
            if connectivity == 4 :  
                F = torch.prod(n[:, :, :, ::2], dim=-1)
            else : 
                F = torch.prod(n, dim=-1)
        else :
            functions = {"minmax" : self.minmax, "drastic" : self.drastic, "bounded" : self.elBoundDiff, "einstein" : self.elEinstein, "hamacher" :self.elHamacher}
            funct = functions[method]
            if connectivity == 4 :
                F = funct(n[:, :, :, ::2],ope = 0)
            else : 
                F = funct(n,ope = 0)
        return F
    

    def forward(self, im, iterations=1, connectivity = 4, method = "product"):
        """
        Inputs :
        - im : input 2D image of shape [batch_size, channels, height, width] or [height, width].
        - iterations : number of times the morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 4 or 8.
        Output : Image after morphological operation
        """
        im = self.test_format(im, connectivity, method) # Check user inputs
        for _ in range(iterations):
            im_padded = F.pad(im, (1, 1, 1, 1), mode='constant', value=1)
            # Unfold the tensor to extract overlapping 3x3 windows
            unf = nn.Unfold((im.shape[2],im.shape[3]), 1, 0, 1)
            unfolded = unf(im_padded) 
            unfolded = unfolded.view(im.shape[0], im.shape[1], -1, unfolded.size(-1))
            # Apply the morphological operation formula to all windows simultaneously
            unfolded = unfolded[:, :, :, (self.indices_list[:, 0] * 3) + self.indices_list[:, 1]]
            output = self.allcondArithm(unfolded, connectivity, method)
            # Adjust the dimensions of output to match the spatial dimensions of im
            output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3])
            # Element-wise multiplication
            im = im * output
        return im


class SoftDilation(nn.Module):
    """
    Class implemented using Pytorch module to perform differentiable soft dilation on 2D input image.
    """
    def __init__(self):
        super(SoftDilation, self).__init__()
        self.indices_list = torch.tensor([
            [0, 1], [0, 2], [1, 2], [2, 2],
            [2, 1], [2, 0], [1, 0], [0, 0], [1,1]
        ], dtype=torch.long)

    def test_format(self, img, connectivity, method):
        """
        Function to check user inputs :
        - Input image shape must either be [batch_size, channels, height, width] or [height, width]. 
        - Input image values must be between 0 and 1.
        - Connectivity represents the sutructuring element of the operation. In 2D, it must be either 4 or 8
        - Method represents the fuzzy logic method used to perform the operation.
        """
        dim = img.dim()
        size = img.size()
        if dim > 4 or dim <2:
            raise Exception(f"Invalid input shape {size}. Expected [batch_size, channels, height, width] or [height, width]. Consider using the 3D version for 3D input images")
        elif dim < 4:
            if dim ==3 :
                # If the input dimension is 3 it might be due to input format [channels, height width]
                if size[0] > 3 : # If this is not likely we raise an exception.
                    raise Exception(f"Ambiguous input shape {size}. Expected [batch_size, channels, height, width] or [height, width].")
            for i in range(4-dim):img = img.unsqueeze(0) 
            print("Image resized to : ", img.size())
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Input image values must be in the range [0, 1].")
        if connectivity != 4 and connectivity !=8:
            raise ValueError("Connectivity should either be 4 or 8")
        if method not in ["product", "multi-linear", "minmax", "drastic", "bounded", "einstein", "hamacher"]:
            raise ValueError("Unvalid thinning method. Choose among 'product', 'multi-linear', 'minmax', 'drastic', 'bounded', 'einstein', 'hamacher'")
        return img
    
    def allcondArithm(self, n, connectivity, method):
        """
        Apply polynomial formula based on the boolean expression that defines a dilation on each 3x3 overlapping squares of the 2D image.
        Inputs : vector of 3x3 overlapping squares n, connectivity (4 or 8) defining the structuring element.
        Output : Returns the new value attributed to each central pixel
        """
        if method == "product" or method == "multi-linear":
            if connectivity == 4 :
                F = 1 - torch.prod(1 - n[:, :, :, ::2], dim=-1)
            else :
                F = 1 - torch.prod(1 - n, dim=-1)
        else :
            functions = {"minmax" : self.minmax, "drastic" : self.drastic, "bounded" : self.elBoundDiff, "einstein" : self.elEinstein, "hamacher" :self.elHamacher}
            funct = functions[method]
            if connectivity == 4:
                F = funct(n[:, :, :, ::2],ope = 1)
            else :
                F = funct(n,ope = 1)
        return F

    def forward(self, im, iterations=1, connectivity = 4, method = "product"):
        """
        Inputs :
        - im : input 2D image of shape [batch_size, channels, height, width] or [height, width].
        - iterations : number of times the morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 4 or 8.
        Output : Image after morphological operation
        """
        im = self.test_format(im, connectivity, method)
        for _ in range(iterations):
            # Unfold the tensor to extract overlapping 3x3 windows
            unf = nn.Unfold((im.shape[2],im.shape[3]), 1, 1, 1)
            unfolded = unf(im) 
            unfolded = unfolded.view(im.shape[0], im.shape[1], -1, unfolded.size(-1))
            # Apply the formula to all windows simultaneously
            unfolded = unfolded[:, :, :, (self.indices_list[:, 0] * 3) + self.indices_list[:, 1]]
            output = self.allcondArithm(unfolded, connectivity, method)
            # Adjust the dimensions of output to match the spatial dimensions of im
            im = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3])
        return im


class SoftClosing(nn.Module):
    """
    Class implemented using Pytorch module to perform differentiable soft closing on 2D input image.
    """
    def __init__(self):
        super(SoftClosing, self).__init__()
        self.dilate = SoftDilation()
        self.erode = SoftErosion()

    def forward(self,input_img, iterations, dilation_connectivity = 4, erosion_connectivity = 4, method = "product"):
        """
        Inputs :
        - im : input 2D image of shape [batch_size, channels, height, width] or [height, width].
        - iterations : number of times each morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 4 or 8.
                         Can define different connectivity values for erosion and dilation
        Output : Image after morphological operation
        """
        output = self.dilate(input_img, iterations, dilation_connectivity, method)
        output = self.erode(output, iterations, erosion_connectivity, method)
        return output
    
class SoftOpening(nn.Module):
    """
    Class implemented using Pytorch module to perform differentiable soft opening on 2D input image.
    """
    def __init__(self):
        super(SoftOpening, self).__init__()
        self.erode = SoftErosion()
        self.dilate = SoftDilation()

    def forward(self,input_img, iterations, dilation_connectivity = 4, erosion_connectivity = 4, method = "product"):
        """
        Inputs :
        - im : input 2D image of shape [batch_size, channels, height, width] or [height, width].
        - iterations : number of times each morphological operation is repeated.
        - connectivity : connectivity representing the structuring element. Should either be 4 or 8.
                         Can define different connectivity values for erosion and dilation
        Output : Image after morphological operation
        """
        output = self.erode(input_img, iterations, erosion_connectivity, method)
        output = self.dilate(output, iterations, dilation_connectivity, method)
        return output


class SoftSkeletonizer(nn.Module):
    """
    Class implemented using Pytorch module to perform differentiable soft skeletonization on 2D input image.
    
    the max_iter input represents the maximum number of times the thinning operation will be repeated. 
    The stop input represents the percentage of pixel values between two thinning iterations that need to be changed to stop the thinning operation.
    """ 
    def __init__(self, max_iter=100, stop=0.02):
        super(SoftSkeletonizer, self).__init__()
        self.maxiter = max_iter
        self.stop = stop
        # Extract ordered index list in each subdirection (North, East, South, West)
        self.indices_list = [self.extract_indices(o) for o in range(4)]
        
    def test_format(self, img, method):
        """
        Function to check user inputs :
        - Input image shape must either be [batch_size, channels, height, width] or [height, width]. 
        - Input image values must be between 0 and 1.
        - Method represents the fuzzy logic method used to perform the operation.
        """
        dim = img.dim()
        size = img.size()
        if dim > 4 or dim <2:
            raise Exception(f"Invalid input shape {size}. Expected [batch_size, channels, height, width] or [height, width]. Consider using the 3D version for 3D input images")
        elif dim < 4:
            if dim ==3 :
                # If the input dimension is 3 it might be due to input format [channels, height width]
                if size[0] > 3 : # If this is not likely we raise an exception.
                    raise Exception(f"Ambiguous input shape {size}. Expected [batch_size, channels, height, width] or [height, width].")
            for i in range(4-dim):img = img.unsqueeze(0) 
            print("Image resized to : ", img.size())
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Input image values must be in the range [0, 1].")
        if method not in ["product", "multi-linear", "minmax", "drastic", "bounded", "einstein", "hamacher"]:
            raise ValueError("Unvalid thinning method. Choose among 'product', 'multi-linear', 'minmax', 'drastic', 'bounded', 'einstein', 'hamacher'")
        return img
    
    def extract_indices(self, o):
        """
        Function to extract extract ordered index list in each subdirection (North, East, South, West)
        """
        indices = torch.tensor([
            [0, 1], [0, 2], [1, 2], [2, 2],
            [2, 1], [2, 0], [1, 0], [0, 0]
        ], dtype=torch.long)
        # Adjust indices based on orientation
        indices = torch.roll(indices, -2 * o, dims=0)
        
        return indices

    def allcondArithm(self, n, method):
        """
        Apply polynomial formula based on the boolean expression that defines a thinning operation on each 3x3 overlapping squares of the 2D image.
        Inputs : n is a vector of 3x3 overlapping squares n, method corresponds to the fuzzy logic method used to perform the thinning operation.
        Output : In binary case returns 0 if the central pixel needs to be changed to 0, returns 1 otherwise.
        """
    
        if method == "product" :
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
        
        
        elif method == "multi-linear" :
            F = (n[:, :, :, 0] - 1)*(3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5] - 3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 7] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 6]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 3] - 3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + 3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5] + 3*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 4] + 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 5]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 5] - 2*n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 6] + n[:, :, :, 1]*n[:, :, :, 2]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 2] - 2*n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5] + 2*n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 4] + n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 3]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 5] - n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 4]*n[:, :, :, 6] - n[:, :, :, 1]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 1]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 1]*n[:, :, :, 6]*n[:, :, :, 7] - 3*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + 3*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] + 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] - 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5] + 3*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6] - 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 7] + 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 4] + 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 7] + n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 5] - 2*n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 6] + n[:, :, :, 2]*n[:, :, :, 3]*n[:, :, :, 7] - n[:, :, :, 2]*n[:, :, :, 3] + 2*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] + n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 5] - 2*n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 6] + n[:, :, :, 2]*n[:, :, :, 4]*n[:, :, :, 7] - n[:, :, :, 2]*n[:, :, :, 4] - n[:, :, :, 2]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 2]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 2]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] + n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 5] - 2*n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 6] + n[:, :, :, 3]*n[:, :, :, 4]*n[:, :, :, 7] - n[:, :, :, 3]*n[:, :, :, 4] - n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + n[:, :, :, 3]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 3]*n[:, :, :, 6]*n[:, :, :, 7] - 2*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] + 2*n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 6] + n[:, :, :, 4]*n[:, :, :, 5]*n[:, :, :, 7] - n[:, :, :, 4]*n[:, :, :, 5] + n[:, :, :, 4]*n[:, :, :, 6]*n[:, :, :, 7] - n[:, :, :, 4]*n[:, :, :, 6] + n[:, :, :, 5]*n[:, :, :, 6]*n[:, :, :, 7] - n[:, :, :, 5]*n[:, :, :, 6] - n[:, :, :, 6]*n[:, :, :, 7])
            F = 1-F

        else : 
            functions = {"minmax" : minmax, "drastic" : drastic, "bounded" : elBoundDiff, "einstein" : elEinstein, "hamacher" :elHamacher}
            funct = functions[method]
            
            F1 = (1 - n[:, :, :, 0])
            

            F2 = funct([
                (1 - n[:, :, :, 1]), (1 - n[:, :, :, 7]),
                funct(
                    [funct([n[:, :, :, 2], (1-n[:, :, :, 3]),n[:, :, :, 4]], ope = 0),
                     funct([(1-n[:, :, :, 2]), n[:, :, :, 3],n[:, :, :, 4]], ope = 0),
                     funct([(1-n[:, :, :, 2]), (1-n[:, :, :, 3]),(1-n[:, :, :, 4])], ope = 0),
                     funct([(n[:, :, :, 2]), n[:, :, :, 3],(1-n[:, :, :, 4])], ope = 0)], ope = 1),
                funct([funct([n[:, :, :, 3],(1 - n[:, :, :, 5])], ope = 0), funct([(1 - n[:, :, :, 3]),n[:, :, :, 5]], ope = 0)], ope = 1),
                funct([funct([n[:, :, :, 3],(1 - n[:, :, :, 6])], ope = 0), funct([(1 - n[:, :, :, 3]),n[:, :, :, 6]], ope = 0)], ope = 1)
                ], ope = 0) 
            F3 = funct([
                funct([funct([n[:, :, :, 1],(1-n[:, :, :, 5])], ope = 0), funct([(1-n[:, :, :, 1]),n[:, :, :, 5]], ope = 0)], ope = 1),
                funct([funct([n[:, :, :, 2],(1-n[:, :, :, 5])], ope = 0), funct([(1 - n[:, :, :, 2]), n[:, :, :, 5]], ope = 0)], ope = 1),
                funct([funct([n[:, :, :, 4], n[:, :, :, 5]], ope = 0), funct([(1 - n[:, :, :, 4]), (1-n[:, :, :, 5])], ope = 0)], ope = 1),
                (1 - n[:, :, :, 6]),(1 - n[:, :, :, 7])], ope = 0) 
            F4 = funct([n[:, :, :, 2],n[:, :, :, 4],(1 - n[:, :, :, 7])], ope = 0)
            F5 = funct([(1 - n[:, :, :, 1]), n[:, :, :, 4], n[:, :, :, 6]], ope = 0)
            F6 = funct([(1 - n[:, :, :, 1]),(1 - n[:, :, :, 2]), (1 - n[:, :, :, 3]), n[:, :, :, 6], n[:, :, :, 7]], ope = 0)
            
            F = 1 - funct([F1, funct([F2,F3,F4,F5,F6], ope = 1)], ope = 0)

        return F
    
    def testchange(self, s1, s2, obj):
        """Function to check if the change between the previous and current image is less than 2% compared to the initial image"""
        rest = abs(s1-s2)
        final = (torch.sum(rest))/obj
        if final <= self.stop:
            self.change = False

    def forward(self, im, method="product"):
        """
        Input :
        - im : input 2D image of shape [batch_size, channels, height, width] or [height, width].
        Output : Image after morphological operation
        """
        im = self.test_format(im, method)
        obj=torch.sum(im)
        self.change = True
        count = 0
        while self.change and count<self.maxiter:
            count +=1
            image = im.clone()
            for o in range(4): # Iterate through all directions
                # Unfold the tensor to extract overlapping 3x3 windows
                unf = nn.Unfold((im.shape[2],im.shape[3]), 1, 1, 1)
                unfolded = unf(im) 
                unfolded = unfolded.view(im.shape[0], im.shape[1], -1, unfolded.size(-1))
                # Apply the formula to all windows simultaneously
                unfolded = unfolded[:, :, :, (self.indices_list[o][:, 0] * 3) + self.indices_list[o][:, 1]]
                output = self.allcondArithm(unfolded, method)
                # Adjust the dimensions of output to match the spatial dimensions of im
                output = output.view(output.size(0), output.size(1), im.shape[2], im.shape[3])
                # Element-wise multiplication
                im = im * output
            # Check if there is more than 2% change between the previous and current image compared to initial image    
            self.testchange(image, im, obj)
        return im   
