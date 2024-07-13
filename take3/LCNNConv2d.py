import torch
import torch.nn as nn
import torch.nn.functional as F

class LCNNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dictionary_size=100, sparsity=0.5):
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
        
        # Lookup weight tensor
        self.lookup_weights = nn.Parameter(torch.Tensor(out_channels, dictionary_size))
        
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dictionary)
        nn.init.kaiming_uniform_(self.lookup_weights)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        print(f"Dictionary shape: {self.dictionary.shape}")
        print(f"Lookup weights shape: {self.lookup_weights.shape}")
        
        self.enforce_sparsity()
        
        input_dictionary = F.conv2d(x, self.dictionary, stride=self.stride, padding=self.padding)
        print(f"Input dictionary shape after convolution: {input_dictionary.shape}")
        
        batch_size, _, output_height, output_width = input_dictionary.shape
        input_dictionary = input_dictionary.view(batch_size, self.dictionary_size, -1)
        input_dictionary = input_dictionary.transpose(1, 2)  # Transpose dimensions 1 and 2
        print(f"Input dictionary shape after reshaping and transposing: {input_dictionary.shape}")
        
        lookup_weights_sparse = self.lookup_weights.to_sparse()
        print(f"Lookup weights sparse shape: {lookup_weights_sparse.shape}")
        
        output = torch.sparse.mm(input_dictionary, lookup_weights_sparse.t())  # Transpose the sparse lookup weights
        print(f"Output shape after sparse matrix multiplication: {output.shape}")
        
        output = output.view(batch_size, output_height, output_width, self.out_channels)
        output = output.permute(0, 3, 1, 2)  # Permute dimensions to (batch_size, out_channels, height, width)
        print(f"Output shape after reshaping and permuting: {output.shape}")
        
        output = output + self.bias.view(1, -1, 1, 1)
        print(f"Final output shape: {output.shape}")
        
        return output

    def enforce_sparsity(self):
        with torch.no_grad():
            # Calculate the number of elements to keep based on sparsity
            num_keep = int(self.lookup_weights.numel() * (1 - self.sparsity))
            
            # Flatten the lookup weights tensor
            flat_weights = self.lookup_weights.view(-1)
            
            # Get the absolute values and indices of the flattened weights
            abs_weights = torch.abs(flat_weights)
            _, indices = torch.sort(abs_weights, descending=True)
            
            # Create a mask to keep the top num_keep elements
            mask = torch.zeros_like(flat_weights, dtype=torch.bool)
            mask[indices[:num_keep]] = True
            
            # Apply the mask to the flattened weights
            flat_weights[~mask] = 0.0
            
            # Reshape the modified weights back to the original shape
            self.lookup_weights.data = flat_weights.view(self.out_channels, self.dictionary_size)
        
        print(f"Lookup weights shape after sparsity enforcement: {self.lookup_weights.shape}")
        print(f"Number of non-zero elements in lookup weights: {self.lookup_weights.nonzero().shape[0]}")
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, '
                f'dictionary_size={self.dictionary_size}, sparsity={self.sparsity}')