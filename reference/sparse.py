import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseConv2D(nn.Module):
    def __init__(self, dense_shape, strides):
        super(SparseConv2D, self).__init__()
        self.dense_shape = dense_shape
        self.strides = strides

    def forward(self, input, weight_indices, weight_values):
        batch_size, in_height, in_width, in_channels = input.shape
        out_channels, kernel_height, kernel_width, _ = self.dense_shape

        # Create a sparse weight tensor
        sparse_weight = torch.sparse_coo_tensor(
            weight_indices.t().long(),
            weight_values,
            size=self.dense_shape
        )

        # Unfold the input tensor
        unfolded = F.unfold(input.permute(0, 3, 1, 2), 
                            kernel_size=(kernel_height, kernel_width), 
                            stride=self.strides)

        # Perform the convolution using sparse matrix multiplication
        out = torch.sparse.mm(sparse_weight.view(out_channels, -1), unfolded)

        # Calculate output dimensions
        out_height = (in_height - kernel_height + self.strides[0]) // self.strides[0]
        out_width = (in_width - kernel_width + self.strides[1]) // self.strides[1]

        # Reshape and permute the output
        return out.view(batch_size, out_channels, out_height, out_width).permute(0, 2, 3, 1)

# Example usage
dense_shape = [32, 3, 3, 64]  # [out_channels, kernel_height, kernel_width, in_channels]
strides = [1, 1]
sparse_conv = SparseConv2D(dense_shape, strides)

# Create dummy input and weights
input = torch.randn(1, 28, 28, 64)
weight_indices = torch.randint(0, 10, (100, 4))  # 100 non-zero elements
weight_values = torch.randn(100)

# Forward pass
output = sparse_conv(input, weight_indices, weight_values)
print(output.shape)