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
        self.register_buffer('lookup_indices', torch.zeros(out_channels, *self.kernel_size, sparsity, dtype=torch.long))
        
        # Lookup coefficient tensor
        self.lookup_coefficients = nn.Parameter(torch.Tensor(out_channels, *self.kernel_size, sparsity))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dictionary)
        nn.init.kaiming_uniform_(self.lookup_coefficients)
        self.lookup_indices.random_(0, self.dictionary_size)

    
    def forward(self, x):
        x = x.to(self.device)
        self.dictionary.data = self.dictionary.data.to(self.device)
        self.lookup_indices = self.lookup_indices.to(self.device)
        self.lookup_coefficients.data = self.lookup_coefficients.data.to(self.device)
        
        # Enforce sparsity before each forward pass
        self.enforce_sparsity()
        
        # Step 1: Convolve the channel weight vectors of the dictionary with the input tensor
        input_dictionary = F.conv2d(x, self.dictionary, stride=self.stride, padding=self.padding)
        
        batch_size, _, output_height, output_width = input_dictionary.shape
        
        # Create sparse weight tensor
        weight_indices = torch.stack([
            self.lookup_indices,
            torch.arange(self.out_channels, device=self.device).view(-1, 1, 1, 1).expand_as(self.lookup_indices),
            torch.arange(self.kernel_size[0], device=self.device).view(1, -1, 1, 1).expand_as(self.lookup_indices),
            torch.arange(self.kernel_size[1], device=self.device).view(1, 1, -1, 1).expand_as(self.lookup_indices)
        ], dim=-1).view(-1, 4)
        
        weight_values = self.lookup_coefficients.view(-1)
        
        sparse_weight = torch.sparse_coo_tensor(
            weight_indices.t().long(),
            weight_values,
            size=(self.dictionary_size, self.out_channels, *self.kernel_size)
        ).to(self.device)
        
        # Perform sparse matrix multiplication
        output = torch.sparse.mm(sparse_weight.view(self.out_channels, -1), 
                                 input_dictionary.view(batch_size, -1).t()).t()
        
        return output.view(batch_size, self.out_channels, output_height, output_width)

    def enforce_sparsity(self):
        with torch.no_grad():
            # Sort coefficients by magnitude
            sorted_coeff, sorted_indices = torch.sort(self.lookup_coefficients.abs(), dim=-1, descending=True)
            
            # Create a mask for the top-k values
            mask = torch.zeros_like(self.lookup_coefficients, dtype=torch.bool)
            mask.scatter_(-1, sorted_indices[..., :self.sparsity], 1)
            
            # Zero out coefficients that are not in the top-k
            self.lookup_coefficients.masked_fill_(~mask, 0)
            
            # Update lookup indices
            new_indices = torch.gather(self.lookup_indices, -1, sorted_indices)
            self.lookup_indices.copy_(new_indices)
