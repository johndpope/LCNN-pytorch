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

cuda_kernel = cp.RawKernel(r'''
extern "C" __global__
void lookup_conv2d_forward(
    const float* input,
    const float* dictionary,
    const long* lookup_indices,
    const float* lookup_coefficients,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int dictionary_size,
    int sparsity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_elements = batch_size * out_channels * height * width;

    for (int i = idx; i < total_elements; i += stride) {
        int w = i % width;
        int h = (i / width) % height;
        int c = (i / (width * height)) % out_channels;
        int b = i / (width * height * out_channels);

        float sum = 0.0f;
        for (int s = 0; s < sparsity; ++s) {
            if (c * sparsity + s >= out_channels * sparsity) continue;  // Bounds check
            int dict_idx = lookup_indices[c * sparsity + s];
            if (dict_idx < 0 || dict_idx >= dictionary_size) continue;  // Bounds check
            float coeff = lookup_coefficients[c * sparsity + s];
            
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = h + kh;
                    int iw = w + kw;
                    if (ih < 0 || ih >= height || iw < 0 || iw >= width) continue;  // Bounds check
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                        int dict_idx_full = (dict_idx * in_channels + ic) * kernel_size * kernel_size + kh * kernel_size + kw;
                        if (input_idx < 0 || input_idx >= batch_size * in_channels * height * width) continue;  // Bounds check
                        if (dict_idx_full < 0 || dict_idx_full >= dictionary_size * in_channels * kernel_size * kernel_size) continue;  // Bounds check
                        sum += input[input_idx] * dictionary[dict_idx_full] * coeff;
                    }
                }
            }
        }
        output[i] = sum;
    }
}
''', 'lookup_conv2d_forward')


cuda_kernel_backward = cp.RawKernel(r'''
extern "C" __global__
void lookup_conv2d_backward(
    const float* grad_output,
    const float* input,
    const float* dictionary,
    const long* lookup_indices,
    const float* lookup_coefficients,
    float* grad_input,
    float* grad_dictionary,
    float* grad_lookup_indices,
    float* grad_lookup_coefficients,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int dictionary_size,
    int sparsity,
    int stride,
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_size = blockDim.x * gridDim.x;

    // Gradient w.r.t. input
    for (int i = idx; i < batch_size * in_channels * in_height * in_width; i += stride_size) {
        int w = i % in_width;
        int h = (i / in_width) % in_height;
        int c = (i / (in_width * in_height)) % in_channels;
        int b = i / (in_width * in_height * in_channels);

        float sum = 0.0f;
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int s = 0; s < sparsity; ++s) {
                int dict_idx = lookup_indices[oc * sparsity + s];
                float coeff = lookup_coefficients[oc * sparsity + s];
                
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int oh = (h + padding - kh) / stride;
                        int ow = (w + padding - kw) / stride;
                        if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                            int grad_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                            int dict_idx_full = (dict_idx * in_channels + c) * kernel_size * kernel_size + kh * kernel_size + kw;
                            sum += grad_output[grad_idx] * dictionary[dict_idx_full] * coeff;
                        }
                    }
                }
            }
        }
        grad_input[i] = sum;
    }

    // Gradient w.r.t. dictionary
    for (int i = idx; i < dictionary_size * in_channels * kernel_size * kernel_size; i += stride_size) {
        int kw = i % kernel_size;
        int kh = (i / kernel_size) % kernel_size;
        int c = (i / (kernel_size * kernel_size)) % in_channels;
        int d = i / (kernel_size * kernel_size * in_channels);

        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int s = 0; s < sparsity; ++s) {
                    if (lookup_indices[oc * sparsity + s] == d) {
                        float coeff = lookup_coefficients[oc * sparsity + s];
                        for (int oh = 0; oh < out_height; ++oh) {
                            for (int ow = 0; ow < out_width; ++ow) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_channels + c) * in_height + ih) * in_width + iw;
                                    int grad_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                                    sum += grad_output[grad_idx] * input[input_idx] * coeff;
                                }
                            }
                        }
                    }
                }
            }
        }
        grad_dictionary[i] = sum;
    }

    // Gradient w.r.t. lookup_coefficients
    for (int i = idx; i < out_channels * sparsity; i += stride_size) {
        int s = i % sparsity;
        int oc = i / sparsity;

        float sum = 0.0f;
        int dict_idx = lookup_indices[i];
        for (int b = 0; b < batch_size; ++b) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float grad_out_val = grad_output[((b * out_channels + oc) * out_height + oh) * out_width + ow];
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                for (int c = 0; c < in_channels; ++c) {
                                    int input_idx = ((b * in_channels + c) * in_height + ih) * in_width + iw;
                                    int dict_idx_full = (dict_idx * in_channels + c) * kernel_size * kernel_size + kh * kernel_size + kw;
                                    sum += grad_out_val * input[input_idx] * dictionary[dict_idx_full];
                                }
                            }
                        }
                    }
                }
            }
        }
        grad_lookup_coefficients[i] = sum;
    }
}
''', 'lookup_conv2d_backward')


class LookupConv2dFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, dictionary, lookup_indices, lookup_coefficients, stride, padding):
        try:
            batch_size, in_channels, in_height, in_width = input.shape
            out_channels, sparsity = lookup_coefficients.shape
            kernel_size = dictionary.shape[2]
            dictionary_size = dictionary.shape[0]
            
            # Compute output dimensions
            out_height = (in_height + 2 * padding - kernel_size) // stride + 1
            out_width = (in_width + 2 * padding - kernel_size) // stride + 1

            # Apply padding to input
            if padding > 0:
                input = F.pad(input, (padding, padding, padding, padding))

            output = torch.zeros(batch_size, out_channels, out_height, out_width, device=input.device)

            threads_per_block = 256
            blocks = (output.numel() + threads_per_block - 1) // threads_per_block

            # print(f"CUDA Kernel Input Shapes: input={input.shape}, dictionary={dictionary.shape}, "
                #   f"lookup_indices={lookup_indices.shape}, lookup_coefficients={lookup_coefficients.shape}")
            # print(f"CUDA Kernel Output Shape: {output.shape}")
            # print(f"CUDA Kernel Launch Config: blocks={blocks}, threads_per_block={threads_per_block}")

            # Check for NaN or Inf values
            if torch.isnan(input).any() or torch.isinf(input).any():
                raise ValueError("Input tensor contains NaN or Inf values")
            if torch.isnan(dictionary).any() or torch.isinf(dictionary).any():
                raise ValueError("Dictionary tensor contains NaN or Inf values")
            if torch.isnan(lookup_coefficients).any() or torch.isinf(lookup_coefficients).any():
                raise ValueError("Lookup coefficients tensor contains NaN or Inf values")

            # Check if lookup_indices are within bounds
            if lookup_indices.min() < 0 or lookup_indices.max() >= dictionary_size:
                raise ValueError(f"Lookup indices out of bounds. Min: {lookup_indices.min()}, Max: {lookup_indices.max()}, Dictionary size: {dictionary_size}")

            cuda_kernel(
                grid=(blocks,),
                block=(threads_per_block,),
                args=(
                    input.data_ptr(),
                    dictionary.data_ptr(),
                    lookup_indices.data_ptr(),
                    lookup_coefficients.data_ptr(),
                    output.data_ptr(),
                    batch_size,
                    in_channels,
                    out_channels,
                    out_height,
                    out_width,
                    kernel_size,
                    dictionary_size,
                    sparsity,
                )
            )

            ctx.save_for_backward(input, dictionary, lookup_indices, lookup_coefficients)
            ctx.stride = stride
            ctx.padding = padding

            return output
        except Exception as e:
            print(f"Error in LookupConv2dFunction.forward: {str(e)}")
            raise

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, dictionary, lookup_indices, lookup_coefficients = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        batch_size, in_channels, in_height, in_width = input.shape
        out_channels, sparsity = lookup_coefficients.shape
        kernel_size = dictionary.shape[2]
        dictionary_size = dictionary.shape[0]
        out_height = grad_output.shape[2]
        out_width = grad_output.shape[3]

        grad_input = torch.zeros_like(input)
        grad_dictionary = torch.zeros_like(dictionary)
        grad_lookup_coefficients = torch.zeros_like(lookup_coefficients)

        threads_per_block = 256
        blocks = (max(input.numel(), dictionary.numel(), lookup_coefficients.numel()) + threads_per_block - 1) // threads_per_block

        cuda_kernel_backward(
            grid=(blocks,),
            block=(threads_per_block,),
            args=(
                grad_output.data_ptr(),
                input.data_ptr(),
                dictionary.data_ptr(),
                lookup_indices.data_ptr(),
                lookup_coefficients.data_ptr(),
                grad_input.data_ptr(),
                grad_dictionary.data_ptr(),
                None,  # grad_lookup_indices (we don't compute gradients for integer indices)
                grad_lookup_coefficients.data_ptr(),
                batch_size,
                in_channels,
                out_channels,
                in_height,
                in_width,
                out_height,
                out_width,
                kernel_size,
                dictionary_size,
                sparsity,
                stride,
                padding
            )
        )

        return grad_input, grad_dictionary, None, grad_lookup_coefficients, None, None


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
        # Change: Use register_buffer for lookup_indices instead of nn.Parameter
        self.register_buffer('lookup_indices', torch.zeros(out_channels, sparsity, dtype=torch.long))
        self.lookup_coefficients = nn.Parameter(torch.Tensor(out_channels, sparsity))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dictionary)
        nn.init.kaiming_uniform_(self.lookup_coefficients)
        # Change: Use in-place operation to set random values for lookup_indices
        self.lookup_indices.random_(0, self.dictionary_size)

    def forward(self, x):
        return LookupConv2dFunction.apply(
            x, self.dictionary, self.lookup_indices, self.lookup_coefficients,
            self.stride, self.padding
        )

# # Example usage
if False:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create sample input and LookupConv2d layer
    batch_size, in_channels, height, width = 1, 3, 32, 32
    out_channels, kernel_size, stride, padding = 64, 3, 1, 1
    dictionary_size, sparsity = 100, 3

    x = torch.randn(batch_size, in_channels, height, width).to(device)
    layer = LookupConv2d(in_channels, out_channels, kernel_size, stride, padding, dictionary_size, sparsity).to(device)

    # Forward pass
    output = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Backward pass (for testing)
    loss = output.sum()
    loss.backward()

    print("Backward pass completed successfully")
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