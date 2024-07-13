import torch
import torch.nn as nn
import torch.nn.functional as F

class LCNNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dictionary_size=100, sparsity=3):
        super(LCNNConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dictionary_size = dictionary_size
        self.sparsity = sparsity

        # Dictionary of channel weight vectors
        self.dictionary = nn.Parameter(torch.Tensor(dictionary_size, in_channels, *self.kernel_size))
        
        # Lookup index tensor
        self.register_buffer('lookup_indices', torch.zeros(out_channels, self.kernel_size[0], self.kernel_size[1], sparsity, dtype=torch.long))
        
        # Lookup coefficient tensor
        self.lookup_coefficients = nn.Parameter(torch.Tensor(out_channels, self.kernel_size[0], self.kernel_size[1], sparsity))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dictionary)
        nn.init.kaiming_uniform_(self.lookup_coefficients)
        self.lookup_indices.random_(0, self.dictionary_size)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        print(f"Dictionary shape: {self.dictionary.shape}")
        print(f"Lookup indices shape: {self.lookup_indices.shape}")
        print(f"Lookup coefficients shape: {self.lookup_coefficients.shape}")
        
        self.enforce_sparsity()
        
        input_dictionary = F.conv2d(x, self.dictionary, stride=self.stride, padding=self.padding)
        print(f"Input dictionary shape after convolution: {input_dictionary.shape}")
        
        batch_size, _, output_height, output_width = input_dictionary.shape
        
        # Reshape lookup indices and coefficients
        lookup_indices_reshaped = self.lookup_indices.view(self.out_channels, -1, self.sparsity)
        lookup_coefficients_reshaped = self.lookup_coefficients.view(self.out_channels, -1, self.sparsity)
        
        print(f"Reshaped lookup indices shape: {lookup_indices_reshaped.shape}")
        print(f"Reshaped lookup coefficients shape: {lookup_coefficients_reshaped.shape}")
        
        # Create sparse indices
        channel_indices = lookup_indices_reshaped.expand(batch_size, -1, -1, -1).reshape(-1)
        out_channel_indices = torch.arange(self.out_channels).repeat_interleave(lookup_indices_reshaped.shape[1] * self.sparsity * batch_size)
        height_indices = torch.arange(output_height).repeat(self.out_channels * lookup_indices_reshaped.shape[1] * self.sparsity).repeat_interleave(output_width)
        width_indices = torch.arange(output_width).repeat(output_height * self.out_channels * lookup_indices_reshaped.shape[1] * self.sparsity)
        
        print(f"Channel indices shape: {channel_indices.shape}")
        print(f"Out channel indices shape: {out_channel_indices.shape}")
        print(f"Height indices shape: {height_indices.shape}")
        print(f"Width indices shape: {width_indices.shape}")
        
        sparse_indices = torch.stack([channel_indices, out_channel_indices, height_indices, width_indices], dim=0)
        print(f"Sparse indices shape: {sparse_indices.shape}")
        
        sparse_values = lookup_coefficients_reshaped.expand(batch_size, -1, -1, -1).reshape(-1)
        print(f"Sparse values shape: {sparse_values.shape}")
        
        sparse_size = (self.dictionary_size, self.out_channels, output_height, output_width)
        print(f"Sparse size: {sparse_size}")
        
        sparse_tensor = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size)
        print(f"Sparse tensor shape: {sparse_tensor.shape}")
        
        output = torch.sparse.mm(sparse_tensor, input_dictionary.reshape(batch_size, -1).t()).t()
        print(f"Output shape after sparse multiplication: {output.shape}")
        
        output = output.reshape(batch_size, self.out_channels, output_height, output_width)
        print(f"Final output shape: {output.shape}")
        
        return output

    def enforce_sparsity(self):
        with torch.no_grad():
            sorted_coeff, sorted_indices = torch.sort(self.lookup_coefficients.abs(), dim=-1, descending=True)
            mask = torch.zeros_like(self.lookup_coefficients, dtype=torch.bool)
            mask.scatter_(-1, sorted_indices[..., :self.sparsity], 1)
            self.lookup_coefficients.masked_fill_(~mask, 0)
            new_indices = torch.gather(self.lookup_indices, -1, sorted_indices)
            self.lookup_indices.copy_(new_indices)

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, '
                f'dictionary_size={self.dictionary_size}, sparsity={self.sparsity}')