import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseSelfAttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_heads=8, sparsity=3):
        super(SparseSelfAttentionConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.num_heads = num_heads
        self.sparsity = sparsity

        # Multi-head self-attention
        self.q_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.k_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.v_conv = nn.Conv2d(in_channels, out_channels, 1)

        # Output projection
        self.out_conv = nn.Conv2d(out_channels, out_channels, 1)

        # Learnable scale factor
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([out_channels // num_heads])))

        # Sparse attention coefficients
        self.attn_coeff = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.attn_coeff)

    def enforce_sparsity(self):
        with torch.no_grad():
            # Sort coefficients by magnitude
            sorted_coeff, sorted_indices = torch.sort(self.attn_coeff.abs(), dim=-1, descending=True)
            
            # Create a mask for the top-k values
            mask = torch.zeros_like(self.attn_coeff, dtype=torch.bool)
            mask.scatter_(-1, sorted_indices[..., :self.sparsity], 1)
            
            # Zero out coefficients that are not in the top-k
            self.attn_coeff.masked_fill_(~mask, 0)

    def forward(self, x):
        self.enforce_sparsity()
        
        batch_size, _, height, width = x.shape

        # Compute Q, K, V
        q = self.q_conv(x).view(batch_size, self.num_heads, self.out_channels // self.num_heads, -1)
        k = self.k_conv(x).view(batch_size, self.num_heads, self.out_channels // self.num_heads, -1)
        v = self.v_conv(x).view(batch_size, self.num_heads, self.out_channels // self.num_heads, -1)

        # Compute attention scores
        attn = torch.matmul(q.transpose(2, 3), k) / self.scale
        
        # Apply sparse attention coefficients
        sparse_attn = torch.matmul(attn, self.attn_coeff[:self.out_channels // self.num_heads, :self.out_channels // self.num_heads].unsqueeze(0).unsqueeze(0))
        sparse_attn = F.softmax(sparse_attn, dim=-1)

        # Apply attention to V
        output = torch.matmul(sparse_attn, v.transpose(2, 3))
        output = output.transpose(2, 3).contiguous().view(batch_size, self.out_channels, height, width)

        # Output projection
        output = self.out_conv(output)

        return output

class LCNNSparseSelfAttention(nn.Module):
    def __init__(self, num_classes):
        super(LCNNSparseSelfAttention, self).__init__()
        self.conv1 = SparseSelfAttentionConv2d(3, 64, kernel_size=11, stride=4, padding=2, sparsity=3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = SparseSelfAttentionConv2d(64, 192, kernel_size=5, padding=2, sparsity=5)
        self.conv3 = SparseSelfAttentionConv2d(192, 384, kernel_size=3, padding=1, sparsity=7)
        self.conv4 = SparseSelfAttentionConv2d(384, 256, kernel_size=3, padding=1, sparsity=7)
        self.conv5 = SparseSelfAttentionConv2d(256, 256, kernel_size=3, padding=1, sparsity=7)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(self.relu(self.conv5(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Example usage
model = LCNNSparseSelfAttention(num_classes=1000)
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(f"Output shape: {output.shape}")