import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import random

from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import cupy as cp


from lookup_conv2d import LookupConv2d


class LookupConv2dNoCuda(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', dictionary_size=100, sparsity=3):
        super(LookupConv2d, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.dictionary_size = dictionary_size
        self.sparsity = sparsity

        self.dictionary = nn.Parameter(torch.Tensor(dictionary_size, in_channels // groups, *kernel_size))
        self.lookup_indices = nn.Parameter(torch.zeros(out_channels, sparsity, dtype=torch.long), requires_grad=False)
        self.lookup_coefficients = nn.Parameter(torch.Tensor(out_channels, sparsity))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dictionary, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.lookup_coefficients, a=np.sqrt(5))
        
        # Use random.randint to generate random indices
        for i in range(self.out_channels):
            for j in range(self.sparsity):
                self.lookup_indices[i, j] = random.randint(0, self.dictionary_size - 1)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dictionary)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            input = F.pad(input, expanded_padding, mode='circular')
            padding = 0
        else:
            padding = self.padding

        S = F.conv2d(input, self.dictionary, None, self.stride, padding, self.dilation, self.groups)
        
        out = torch.zeros(input.size(0), self.out_channels, 
                          ((input.size(2) + 2*padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1,
                          ((input.size(3) + 2*padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1,
                          device=input.device)
        
        for i in range(self.out_channels):
            for j in range(self.sparsity):
                out[:, i] += S[:, self.lookup_indices[i, j]] * self.lookup_coefficients[i, j]
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        
        return out
class LCNNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dictionary_size=100, sparsity=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dictionary_size = dictionary_size
        self.sparsity = sparsity
        
        # Dictionary
        self.D = nn.Parameter(torch.Tensor(dictionary_size, in_channels, 1, 1))
        
        # Sparse tensor P 
        self.P = nn.Parameter(torch.Tensor(out_channels, dictionary_size, *kernel_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.D)
        nn.init.kaiming_uniform_(self.P)
        
    def forward(self, x):
        # Convolve input with dictionary
        S = F.conv2d(x, self.D)
        
        # Lookup and combine
        out = F.conv2d(S, self.P, stride=self.stride, padding=self.padding)
        
        return out
        
    def enforce_sparsity(self):
        with torch.no_grad():
            self.P.data = self.get_sparse_weights(self.P.data)
            
    def get_sparse_weights(self, weights):
        abs_weights = torch.abs(weights)
        k_largest = torch.topk(abs_weights.view(weights.size(0), -1), 
                               k=self.sparsity, dim=1)[1]
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.view(weights.size(0), -1).scatter_(1, k_largest, 
                            weights.view(weights.size(0), -1).gather(1, k_largest))
        return sparse_weights

def profile_conv(conv_layer, input_tensor, num_iterations=100):
    try:
        conv_layer.cuda()
        input_tensor = input_tensor.cuda()
        
        # Warm-up
        for _ in range(10):
            _ = conv_layer(input_tensor)
        
        torch.cuda.synchronize()
        
        # Speed profiling
        start_time = time.time()
        for _ in range(num_iterations):
            _ = conv_layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        
        # Memory profiling
        torch.cuda.reset_peak_memory_stats()
        _ = conv_layer(input_tensor)
        torch.cuda.synchronize()
        memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
        
        return avg_time, memory_usage
    except Exception as e:
        print(f"Error in profile_conv: {str(e)}")
        return None, None

def compare_conv_layers(in_channels, out_channels, kernel_size, input_size, dictionary_size=100, sparsity=3):
    try:
        # Create layers
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        lookup_conv = LookupConv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, dictionary_size=dictionary_size, sparsity=sparsity)
        # lookup_conv = LookupConv2dNoCuda(in_channels, out_channels, kernel_size, padding=kernel_size//2, dictionary_size=dictionary_size, sparsity=sparsity)
        
        #lookup_conv = LCNNConv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, dictionary_size=dictionary_size, sparsity=sparsity)
        
        

        # Create input tensor
        x = torch.randn(1, in_channels, input_size, input_size)
        
        # Profile Conv2d
        conv_time, conv_memory = profile_conv(conv, x)
        
        # Profile LookupConv2d
        lookup_time, lookup_memory = profile_conv(lookup_conv, x)
        
        if conv_time is not None and lookup_time is not None:
            print(f"Conv2d - Avg time: {conv_time*1000:.2f} ms, Memory usage: {conv_memory:.2f} MB")
            print(f"LookupConv2d - Avg time: {lookup_time*1000:.2f} ms, Memory usage: {lookup_memory:.2f} MB")
            print(f"Speed difference: {conv_time/lookup_time:.2f}x")
            print(f"Memory difference: {conv_memory/lookup_memory:.2f}x")

        # Detailed profiling using torch.profiler
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("Conv2d"):
                _ = conv(x.cuda())
            with record_function("LookupConv2d"):
                _ = lookup_conv(x.cuda())

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    except Exception as e:
        print(f"Error in compare_conv_layers: {str(e)}")

if __name__ == '__main__':
    # Example usage
    compare_conv_layers(in_channels=64, out_channels=128, kernel_size=3, input_size=224, dictionary_size=10, sparsity=1)