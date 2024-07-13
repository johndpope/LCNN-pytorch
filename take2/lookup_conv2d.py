import torch
import torch.nn as nn
from torch.autograd import Function
import sparse_conv2d_cuda

class SparseConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, dictionary, lookup_indices, lookup_coefficients, stride, padding):
        ctx.save_for_backward(input, dictionary, lookup_indices, lookup_coefficients)
        ctx.stride = stride
        ctx.padding = padding
        return sparse_conv2d_cuda.forward(input, dictionary, lookup_indices, lookup_coefficients, stride, padding)

    @staticmethod
    def backward(ctx, grad_output):
        input, dictionary, lookup_indices, lookup_coefficients = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        
        grad_input, grad_dictionary, grad_coefficients = sparse_conv2d_cuda.backward(
            grad_output.contiguous(), input, dictionary, lookup_indices, lookup_coefficients, stride, padding)
        
        return grad_input, grad_dictionary, None, grad_coefficients, None, None

class LookupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dictionary_size=100, sparsity=3):
        super(LookupConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dictionary_size = dictionary_size
        self.sparsity = sparsity
        
        self.dictionary = nn.Parameter(torch.Tensor(dictionary_size, in_channels, kernel_size, kernel_size))
        self.register_buffer('lookup_indices', torch.zeros(out_channels, sparsity, dtype=torch.long))
        self.lookup_coefficients = nn.Parameter(torch.Tensor(out_channels, sparsity))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dictionary)
        nn.init.kaiming_uniform_(self.lookup_coefficients)
        self.lookup_indices.random_(0, self.dictionary_size)
    
    def forward(self, x):
        # Enforce sparsity before each forward pass
        self.enforce_sparsity()
        return SparseConv2dFunction.apply(
            x, self.dictionary, self.lookup_indices, self.lookup_coefficients,
            self.stride, self.padding)

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, '
                f'dictionary_size={self.dictionary_size}, sparsity={self.sparsity}')
    

    '''
    This enforce_sparsity method does the following:

    It sorts the absolute values of the lookup coefficients.
    It creates a mask for the top-k (where k is the sparsity) values.
    It zeros out the coefficients that are not in the top-k.
    It updates the lookup indices to match the new order of coefficients.

    By calling this method before each forward pass, we ensure that the sparsity constraint is always maintained.

'''
    def enforce_sparsity(self):
        with torch.no_grad():
            # Sort coefficients by magnitude
            sorted_coeff, sorted_indices = torch.sort(self.lookup_coefficients.abs(), dim=1, descending=True)
            
            # Create a mask for the top-k values
            mask = torch.zeros_like(self.lookup_coefficients, dtype=torch.bool)
            mask.scatter_(1, sorted_indices[:, :self.sparsity], 1)
            
            # Zero out coefficients that are not in the top-k
            self.lookup_coefficients.masked_fill_(~mask, 0)
            
            # Update lookup indices
            new_indices = torch.gather(self.lookup_indices, 1, sorted_indices)
            self.lookup_indices.copy_(new_indices)

