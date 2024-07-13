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


template <typename scalar_t>
__global__ void sparse_conv2d_backward_input_kernel(
    const scalar_t* grad_output,
    const scalar_t* dictionary,
    const int64_t* lookup_indices,
    const scalar_t* lookup_coefficients,
    scalar_t* grad_input,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int dictionary_size, int sparsity) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_channels * in_height * in_width) return;

    int w_in = idx % in_width;
    int h_in = (idx / in_width) % in_height;
    int c_in = (idx / (in_width * in_height)) % in_channels;
    int n = idx / (in_width * in_height * in_channels);

    scalar_t sum = 0;

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int s = 0; s < sparsity; ++s) {
            int dict_idx = lookup_indices[oc * sparsity + s];
            scalar_t coeff = lookup_coefficients[oc * sparsity + s];

            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_out = (h_in + padding - kh) / stride;
                    int w_out = (w_in + padding - kw) / stride;
                    
                    if (h_out >= 0 && h_out < out_height && w_out >= 0 && w_out < out_width &&
                        (h_in + padding - kh) % stride == 0 && (w_in + padding - kw) % stride == 0) {
                        int grad_output_idx = ((n * out_channels + oc) * out_height + h_out) * out_width + w_out;
                        int dict_idx_full = ((dict_idx * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                        
                        sum += grad_output[grad_output_idx] * dictionary[dict_idx_full] * coeff;
                    }
                }
            }
        }
    }

    grad_input[idx] = sum;
}

template <typename scalar_t>
__global__ void sparse_conv2d_backward_dictionary_kernel(
    const scalar_t* grad_output,
    const scalar_t* input,
    const int64_t* lookup_indices,
    const scalar_t* lookup_coefficients,
    scalar_t* grad_dictionary,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int dictionary_size, int sparsity) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dictionary_size * in_channels * kernel_size * kernel_size) return;

    int kw = idx % kernel_size;
    int kh = (idx / kernel_size) % kernel_size;
    int c_in = (idx / (kernel_size * kernel_size)) % in_channels;
    int dict_idx = idx / (in_channels * kernel_size * kernel_size);

    scalar_t sum = 0;

    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int s = 0; s < sparsity; ++s) {
                if (lookup_indices[oc * sparsity + s] == dict_idx) {
                    scalar_t coeff = lookup_coefficients[oc * sparsity + s];
                    
                    for (int h_out = 0; h_out < out_height; ++h_out) {
                        for (int w_out = 0; w_out < out_width; ++w_out) {
                            int h_in = h_out * stride - padding + kh;
                            int w_in = w_out * stride - padding + kw;
                            
                            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                                int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                                int grad_output_idx = ((n * out_channels + oc) * out_height + h_out) * out_width + w_out;
                                
                                sum += grad_output[grad_output_idx] * input[input_idx] * coeff;
                            }
                        }
                    }
                }
            }
        }
    }

    grad_dictionary[idx] = sum;
}

template <typename scalar_t>
__global__ void sparse_conv2d_backward_coefficients_kernel(
    const scalar_t* grad_output,
    const scalar_t* input,
    const scalar_t* dictionary,
    const int64_t* lookup_indices,
    scalar_t* grad_coefficients,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding,
    int dictionary_size, int sparsity) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_channels * sparsity) return;

    int s = idx % sparsity;
    int oc = idx / sparsity;

    int dict_idx = lookup_indices[idx];
    scalar_t sum = 0;

    for (int n = 0; n < batch_size; ++n) {
        for (int h_out = 0; h_out < out_height; ++h_out) {
            for (int w_out = 0; w_out < out_width; ++w_out) {
                int grad_output_idx = ((n * out_channels + oc) * out_height + h_out) * out_width + w_out;
                scalar_t grad = grad_output[grad_output_idx];

                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int h_in = h_out * stride - padding + kh;
                            int w_in = w_out * stride - padding + kw;
                            
                            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                                int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                                int dict_idx_full = ((dict_idx * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                                
                                sum += grad * input[input_idx] * dictionary[dict_idx_full];
                            }
                        }
                    }
                }
            }
        }
    }

    grad_coefficients[idx] = sum;
}

// Forward declaration of the CUDA functions
torch::Tensor sparse_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor dictionary,
    torch::Tensor lookup_indices,
    torch::Tensor lookup_coefficients,
    int stride, int padding);

std::vector<torch::Tensor> sparse_conv2d_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor dictionary,
    torch::Tensor lookup_indices,
    torch::Tensor lookup_coefficients,
    int stride, int padding);


std::vector<torch::Tensor> sparse_conv2d_backward_cuda(
    torch::Tensor grad_output,
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

    const auto out_height = grad_output.size(2);
    const auto out_width = grad_output.size(3);

    auto grad_input = torch::zeros_like(input);
    auto grad_dictionary = torch::zeros_like(dictionary);
    auto grad_coefficients = torch::zeros_like(lookup_coefficients);

    const int threads = 1024;

    const int blocks_input = (batch_size * in_channels * in_height * in_width + threads - 1) / threads;
    const int blocks_dictionary = (dictionary_size * in_channels * kernel_size * kernel_size + threads - 1) / threads;
    const int blocks_coefficients = (out_channels * sparsity + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "sparse_conv2d_backward_cuda", ([&] {
        sparse_conv2d_backward_input_kernel<scalar_t><<<blocks_input, threads>>>(
            grad_output.data<scalar_t>(),
            dictionary.data<scalar_t>(),
            lookup_indices.data<int64_t>(),
            lookup_coefficients.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            kernel_size, stride, padding,
            dictionary_size, sparsity
        );

        sparse_conv2d_backward_dictionary_kernel<scalar_t><<<blocks_dictionary, threads>>>(
            grad_output.data<scalar_t>(),
            input.data<scalar_t>(),
            lookup_indices.data<int64_t>(),
            lookup_coefficients.data<scalar_t>(),
            grad_dictionary.data<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            kernel_size, stride, padding,
            dictionary_size, sparsity
        );

        sparse_conv2d_backward_coefficients_kernel<scalar_t><<<blocks_coefficients, threads>>>(
            grad_output.data<scalar_t>(),
            input.data<scalar_t>(),
            dictionary.data<scalar_t>(),
            lookup_indices.data<int64_t>(),
            grad_coefficients.data<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            kernel_size, stride, padding,
            dictionary_size, sparsity
        );
    }));

    return {grad_input, grad_dictionary, grad_coefficients};
}