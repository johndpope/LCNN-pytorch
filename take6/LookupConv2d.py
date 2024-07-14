import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import math

sparse_conv2d_cpp = load(
    name="sparse_conv2d",
    sources=["sparse_conv2d.cpp", "sparse_conv2d_cuda.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True
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
        
        print(f"LookupAlignConvolution2d initialized with:")
        print(f"  in_channels: {in_channels}")
        print(f"  out_channels: {out_channels}")
        print(f"  kernel_size: {self.kernel_size}")
        print(f"  stride: {self.stride}")
        print(f"  padding: {self.padding}")
        print(f"  use_cuda: {self.use_cuda}")

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        print(f"Forward pass input shape: {input.shape}")
        print(f"Input device: {input.device}")
        assert input.dim() == 4, f"Expected 4D input (got {input.dim()}D input)"
        assert input.size(3) == self.in_channels, f"Expected {self.in_channels} input channels, got {input.size(3)}"
        
        # Move weight and bias to the same device as input
        self.weight = self.weight.to(input.device)
        if self.bias is not None:
            self.bias = self.bias.to(input.device)
        
        # Apply sparsity threshold
        weight = torch.where(torch.abs(self.weight) < self.sparse_th, torch.zeros_like(self.weight), self.weight)
        
        # Convert dense weight to sparse representation
        indices = torch.nonzero(weight, as_tuple=False)
        values = weight[weight != 0]
        
        dense_shape = list(weight.shape)
        print(f"Dense weight shape: {dense_shape}")
        print(f"Sparse weight indices shape: {indices.shape}")
        print(f"Sparse weight values shape: {values.shape}")
        
        # Calculate output dimensions
        in_height, in_width = input.shape[1], input.shape[2]
        out_height = (in_height - self.kernel_size[0] + self.stride[0]) // self.stride[0]
        out_width = (in_width - self.kernel_size[1] + self.stride[1]) // self.stride[1]
        
        print(f"Calculated output dimensions: {out_height}x{out_width}")
        
        # Check for valid output dimensions
        if out_height <= 0 or out_width <= 0:
            raise ValueError(f"Invalid output dimensions: {out_height}x{out_width}. "
                             f"Input: {in_height}x{in_width}, Kernel: {self.kernel_size}, Stride: {self.stride}")
        
        try:
            if self.use_cuda and input.is_cuda:
                print("Using CUDA implementation")
                output = sparse_conv2d_cpp.forward_cuda(input, indices, values, dense_shape, list(self.stride))
            else:
                print("Using CPU implementation")
                output = sparse_conv2d_cpp.forward_cpu(input, indices, values, dense_shape, list(self.stride))
        except RuntimeError as e:
            print(f"Error in {'CUDA' if self.use_cuda else 'CPU'} implementation:")
            print(str(e))
            print(f"Input shape: {input.shape}")
            print(f"Dense shape: {dense_shape}")
            print(f"Stride: {self.stride}")
            raise

        print(f"Output shape after convolution: {output.shape}")
        print(f"Output device: {output.device}")
        
        if self.bias is not None:
            print(f"Bias device: {self.bias.device}")
            output += self.bias.view(1, 1, 1, -1)
        
        return output


def lookup_conv2d(x, num_outputs, kernel_size, stride, dict_size, padding=1,
                  param_lambda=0.3, initial_sparsity=None, activation_fn=None, use_cuda=False):
    print(f"\nlookup_conv2d called with:")
    print(f"  Input shape: {x.shape}")
    print(f"  Input device: {x.device}")
    print(f"  num_outputs: {num_outputs}")
    print(f"  kernel_size: {kernel_size}")
    print(f"  stride: {stride}")
    print(f"  dict_size: {dict_size}")
    print(f"  padding: {padding}")
    print(f"  use_cuda: {use_cuda}")
    
    if not initial_sparsity:
        initial_sparsity = 0.5
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    
    sparse_th = initial_sparsity / math.sqrt(kernel_size[0] * kernel_size[1] * dict_size)
    print(f"Calculated sparse_th: {sparse_th}")
    
    # Padding
    x = F.pad(x, (padding, padding, padding, padding))
    print(f"Shape after padding: {x.shape}")
    
    # LookupAlignConvolution2d
    conv = LookupAlignConvolution2d(x.size(3), num_outputs, kernel_size, stride=stride,
                                    padding='valid', param_lambda=param_lambda * sparse_th,
                                    sparse_th=sparse_th, bias=True, use_cuda=use_cuda)
    
    # Move the conv module to the same device as the input
    conv = conv.to(x.device)
    
    output = conv(x)
    
    print(f"Shape after convolution: {output.shape}")
    print(f"Output device: {output.device}")
    
    if activation_fn:
        output = activation_fn(output)
        print(f"Shape after activation: {output.shape}")
    
    return output

# Usage example:
# x = torch.randn(1, 32, 32, 3).cuda()  # Move to GPU if using CUDA
# print("\nInput tensor:")
# print(f"  Shape: {x.shape}")
# print(f"  Device: {x.device}")
# print(f"  dtype: {x.dtype}")

# out = lookup_conv2d(x, num_outputs=64, kernel_size=3, stride=1, dict_size=256, use_cuda=True)
# print(f"\nFinal output shape: {out.shape}")
# print(f"Final output device: {out.device}")