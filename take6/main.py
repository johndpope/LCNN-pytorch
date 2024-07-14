import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import math
import torch.nn.functional as F

sparse_conv2d_cpp = load(
    name="sparse_conv2d",
    sources=["sparse_conv2d.cpp", "sparse_conv2d_cuda.cu"],
    extra_cuda_cflags=["-O2"]
)

class LookupAlignConvolution2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid',
                 dilation=1, groups=1, bias=True, param_lambda=1.0, sparse_th=0.01, use_cuda=False):
        super(LookupAlignConvolution2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.param_lambda = param_lambda
        self.sparse_th = sparse_th
        self.use_cuda = use_cuda

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Apply sparsity threshold
        weight = torch.where(torch.abs(self.weight) < self.sparse_th, torch.zeros_like(self.weight), self.weight)
        
        # Convert dense weight to sparse representation
        indices = torch.nonzero(weight, as_tuple=False)
        values = weight[weight != 0]
        
        dense_shape = list(weight.shape)
        
        if self.use_cuda and input.is_cuda:
            output = sparse_conv2d_cpp.forward_cuda(input, indices, values, dense_shape, list(self.stride))
        else:
            output = sparse_conv2d_cpp.forward_cpu(input, indices, values, dense_shape, list(self.stride))
        
        if self.bias is not None:
            output += self.bias.view(1, 1, 1, -1)
        
        return output

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 'valid':
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.param_lambda != 1.0:
            s += ', param_lambda={param_lambda}'
        if self.sparse_th != 0.01:
            s += ', sparse_th={sparse_th}'
        if self.use_cuda:
            s += ', use_cuda=True'
        return s.format(**self.__dict__)

def lookup_conv2d(x, num_outputs, kernel_size, stride, dict_size, padding=1,
                  param_lambda=0.3, initial_sparsity=None, activation_fn=None, use_cuda=False):
    if not initial_sparsity:
        initial_sparsity = 0.5
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    
    sparse_th = initial_sparsity / math.sqrt(kernel_size[0] * kernel_size[1] * dict_size)
    
    # Padding
    x = F.pad(x, (padding, padding, padding, padding))
    
    # LookupAlignConvolution2d
    conv = LookupAlignConvolution2d(x.size(3), num_outputs, kernel_size, stride=stride,
                                    padding='valid', param_lambda=param_lambda * sparse_th,
                                    sparse_th=sparse_th, bias=True, use_cuda=use_cuda)
    output = conv(x)
    
    if activation_fn:
        output = activation_fn(output)
    
    return output

# Usage example:
x = torch.randn(1, 32, 32, 3).cuda()  # Move to GPU if using CUDA
out = lookup_conv2d(x, num_outputs=64, kernel_size=3, stride=1, dict_size=256, use_cuda=True)
print(out.shape)