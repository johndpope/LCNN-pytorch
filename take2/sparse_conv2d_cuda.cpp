// sparse_conv2d_cuda.cpp
#include <torch/extension.h>

torch::Tensor sparse_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor dictionary,
    torch::Tensor lookup_indices,
    torch::Tensor lookup_coefficients,
    int stride, int padding);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor sparse_conv2d_forward(
    torch::Tensor input,
    torch::Tensor dictionary,
    torch::Tensor lookup_indices,
    torch::Tensor lookup_coefficients,
    int stride, int padding) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(dictionary);
    CHECK_INPUT(lookup_indices);
    CHECK_INPUT(lookup_coefficients);

    return sparse_conv2d_forward_cuda(input, dictionary, lookup_indices, lookup_coefficients, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_conv2d_forward, "Sparse Conv2d forward (CUDA)");
}