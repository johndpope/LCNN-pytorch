#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void sparse_conv2d_cuda_kernel(
    const float* input,
    const int64_t* weight_indices,
    const float* weight_values,
    float* output,
    int64_t batch_size,
    int64_t in_height,
    int64_t in_width,
    int64_t in_channels,
    int64_t out_height,
    int64_t out_width,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t num_weights) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_weights) return;

    int64_t sparse_oc = weight_indices[index * 2];
    int64_t sparse_p = weight_indices[index * 2 + 1];
    int64_t sparse_ic = sparse_p / (kernel_size * kernel_size);
    int64_t sparse_ix = (sparse_p % (kernel_size * kernel_size)) / kernel_size;
    int64_t sparse_iy = (sparse_p % (kernel_size * kernel_size)) % kernel_size;

    float sparse_v = weight_values[index];

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t oh = 0; oh < out_height; ++oh) {
            for (int64_t ow = 0; ow < out_width; ++ow) {
                int64_t ih = oh * stride + sparse_ix;
                int64_t iw = ow * stride + sparse_iy;
                
                float input_val = input[((b * in_height + ih) * in_width + iw) * in_channels + sparse_ic];
                atomicAdd(&output[((b * out_height + oh) * out_width + ow) * out_channels + sparse_oc], input_val * sparse_v);
            }
        }
    }
}

// CUDA implementation
torch::Tensor sparse_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight_indices,
    torch::Tensor weight_values,
    std::vector<int64_t> dense_shape,
    std::vector<int64_t> strides) {
    
    auto batch_size = input.size(0);
    auto in_height = input.size(1);
    auto in_width = input.size(2);
    auto in_channels = input.size(3);

    auto out_height = (in_height - dense_shape[1] + strides[0]) / strides[0];
    auto out_width = (in_width - dense_shape[2] + strides[1]) / strides[1];
    auto out_channels = dense_shape[0];

    std::cout << "CUDA Implementation:" << std::endl;
    std::cout << "Input shape: [" << batch_size << ", " << in_height << ", " << in_width << ", " << in_channels << "]" << std::endl;
    std::cout << "Dense shape: [" << dense_shape[0] << ", " << dense_shape[1] << ", " << dense_shape[2] << ", " << dense_shape[3] << "]" << std::endl;
    std::cout << "Strides: [" << strides[0] << ", " << strides[1] << "]" << std::endl;
    std::cout << "Calculated output shape: [" << batch_size << ", " << out_height << ", " << out_width << ", " << out_channels << "]" << std::endl;

    if (out_height <= 0 || out_width <= 0) {
        throw std::runtime_error("Invalid output dimensions: " + std::to_string(out_height) + "x" + std::to_string(out_width));
    }

    auto output = torch::zeros({batch_size, out_height, out_width, out_channels}, input.options());

    const int threads = 1024;
    const int blocks = (weight_indices.size(0) + threads - 1) / threads;

    sparse_conv2d_cuda_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight_indices.data_ptr<int64_t>(),
        weight_values.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_height,
        in_width,
        in_channels,
        out_height,
        out_width,
        out_channels,
        dense_shape[1],  // kernel size
        strides[0],
        weight_indices.size(0)
    );

    return output;
}