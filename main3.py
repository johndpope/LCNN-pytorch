import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import random
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms


class LCNNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dict_size=100, sparsity=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dict_size = dict_size
        self.sparsity = sparsity
        
        # Dictionary
        self.D = nn.Parameter(torch.Tensor(dict_size, in_channels, 1, 1))
        
        # Sparse tensor P 
        self.P = nn.Parameter(torch.Tensor(out_channels, dict_size, *kernel_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.D)
        nn.init.kaiming_uniform_(self.P)
        
    def forward(self, x):
        # Convolve input with dictionary
        S = F.conv2d(x, self.D)
        
        # Lookup and combine
        out = F.conv2d(S, self.P, stride=self.stride, padding=self.padding)
        
        return out
        
    def enforce_sparsity(self):
        with torch.no_grad():
            self.P.data = self.get_sparse_weights(self.P.data)
            
    def get_sparse_weights(self, weights):
        abs_weights = torch.abs(weights)
        k_largest = torch.topk(abs_weights.view(weights.size(0), -1), 
                               k=self.sparsity, dim=1)[1]
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.view(weights.size(0), -1).scatter_(1, k_largest, 
                            weights.view(weights.size(0), -1).gather(1, k_largest))
        return sparse_weights
    
class LCNNAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            LCNNConv2d(3, 64, kernel_size=11, stride=4, padding=2, dict_size=100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LCNNConv2d(64, 192, kernel_size=5, padding=2, dict_size=100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LCNNConv2d(192, 384, kernel_size=3, padding=1, dict_size=100),
            nn.ReLU(inplace=True),
            LCNNConv2d(384, 256, kernel_size=3, padding=1, dict_size=100),
            nn.ReLU(inplace=True),
            LCNNConv2d(256, 256, kernel_size=3, padding=1, dict_size=100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            LCNNConv2d(256, 4096, kernel_size=6, dict_size=512),
            nn.ReLU(inplace=True),
            LCNNConv2d(4096, 4096, kernel_size=1, dict_size=512),
            nn.ReLU(inplace=True),
            LCNNConv2d(4096, num_classes, kernel_size=1, dict_size=512),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)
    

def few_shot_finetune(base_model, novel_data, num_novel_classes, num_shots):
    # Replace last layer
    base_model.classifier[-1] = LCNNConv2d(4096, num_novel_classes, 
                                           kernel_size=1, dict_size=512)
    
    # Freeze all layers except last
    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.classifier[-1].parameters():
        param.requires_grad = True
        
    # Fine-tune
    optimizer = torch.optim.SGD(base_model.classifier[-1].parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(finetune_epochs):
        for inputs, labels in novel_data:
            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            
            base_model.classifier[-1].enforce_sparsity()
            
    return base_model

model = LCNNAlexNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()




# Load the dataset
dataset = load_dataset("lansinuote/gen.1.celeba")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a standard size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Create a function to apply transformations
def transform_images(examples):
    examples['pixel_values'] = [transform(image.convert('RGB')) for image in examples['image']]
    return examples

# Apply the transformations to the dataset
dataset = dataset.map(transform_images, batched=True, remove_columns=['image'])

# Set the format of the dataset to PyTorch
dataset.set_format(type='torch', columns=['pixel_values', 'attributes'])

# Create DataLoaders
batch_size = 32

train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset['validation'], batch_size=batch_size)
test_dataloader = DataLoader(dataset['test'], batch_size=batch_size)

# Now you can use these DataLoaders in your training loop
# For example:


for epoch in range(100):
    for batch in train_dataloader:
        inputs = batch['pixel_values']
        labels = batch['attributes']

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Enforce sparsity
        for module in model.modules():
            if isinstance(module, LCNNConv2d):
                module.enforce_sparsity()