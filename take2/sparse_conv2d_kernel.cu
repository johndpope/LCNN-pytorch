// sparse_conv2d_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sparse_conv2d_forward_kernel(
    const scalar_t* input,
    const scalar_t* dictionary,
    const int64_t* lookup_indices,
    const scalar_t* lookup_coefficients,
    scalar_t* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int dictionary_size, int sparsity) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_height * out_width) return;

    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int c_out = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);

    scalar_t sum = 0;

    for (int s = 0; s < sparsity; ++s) {
        int dict_idx = lookup_indices[c_out * sparsity + s];
        scalar_t coeff = lookup_coefficients[c_out * sparsity + s];

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;
                    
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = ((n * in_channels + ic) * in_height + h_in) * in_width + w_in;
                        int dict_idx_full = ((dict_idx * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        
                        sum += input[input_idx] * dictionary[dict_idx_full] * coeff;
                    }
                }
            }
        }
    }

    output[idx] = sum;
}

torch::Tensor sparse_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor dictionary,
    torch::Tensor lookup_indices,
    torch::Tensor lookup_coefficients,
    int stride, int padding) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);
    const auto kernel_size = dictionary.size(2);
    const auto out_channels = lookup_indices.size(0);
    const auto dictionary_size = dictionary.size(0);
    const auto sparsity = lookup_indices.size(1);

    const auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int threads = 1024;
    const int blocks = (batch_size * out_channels * out_height * out_width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "sparse_conv2d_forward_cuda", ([&] {
        sparse_conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            dictionary.data<scalar_t>(),
            lookup_indices.data<int64_t>(),
            lookup_coefficients.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            kernel_size, stride, padding,
            dictionary_size, sparsity
        );
    }));

    return output;
}

