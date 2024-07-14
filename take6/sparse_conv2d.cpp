#include <torch/extension.h>
#include <vector>

// CPU implementation
torch::Tensor sparse_conv2d_cpu(
    torch::Tensor input,
    torch::Tensor weight_indices,
    torch::Tensor weight_values,
    std::vector<int64_t> dense_shape,
    std::vector<int64_t> strides) {
    
    auto batch_size = input.size(0);
    auto in_height = input.size(1);
    auto in_width = input.size(2);
    auto in_channels = input.size(3);

    auto out_height = (in_height - dense_shape[0] + strides[0]) / strides[0];
    auto out_width = (in_width - dense_shape[1] + strides[1]) / strides[1];
    auto out_channels = dense_shape[3];

    auto output = torch::zeros({batch_size, out_height, out_width, out_channels}, input.options());

    auto input_accessor = input.accessor<float, 4>();
    auto indices_accessor = weight_indices.accessor<int64_t, 2>();
    auto values_accessor = weight_values.accessor<float, 1>();
    auto output_accessor = output.accessor<float, 4>();

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int64_t sparse_idx = 0; sparse_idx < weight_indices.size(0); ++sparse_idx) {
            int64_t sparse_oc = indices_accessor[sparse_idx][0];
            int64_t sparse_p = indices_accessor[sparse_idx][1];
            int64_t sparse_ic = sparse_p / (dense_shape[1] * dense_shape[2]);
            int64_t sparse_ix = (sparse_p % (dense_shape[1] * dense_shape[2])) / dense_shape[1];
            int64_t sparse_iy = (sparse_p % (dense_shape[1] * dense_shape[2])) % dense_shape[1];

            float sparse_v = values_accessor[sparse_idx];

            for (int64_t row = 0; row < in_height; row += strides[0]) {
                int64_t out_row = row / strides[0];
                for (int64_t col = 0; col < in_width; col += strides[1]) {
                    int64_t out_col = col / strides[1];

                    output_accessor[batch_idx][out_row][out_col][sparse_oc] +=
                        input_accessor[batch_idx][row + sparse_ix][col + sparse_iy][sparse_ic] * sparse_v;
                }
            }
        }
    }

    return output;
}

// CUDA forward declaration
torch::Tensor sparse_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight_indices,
    torch::Tensor weight_values,
    std::vector<int64_t> dense_shape,
    std::vector<int64_t> strides);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &sparse_conv2d_cpu, "SparseConv2D forward (CPU)");
    m.def("forward_cuda", &sparse_conv2d_cuda, "SparseConv2D forward (CUDA)");
}