import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from torchvision import transforms
import os
from tqdm import tqdm

from lookup_conv2d import LookupConv2d

class MyModel(nn.Module):
    def __init__(self, num_attributes):
        super(MyModel, self).__init__()
        self.conv1 = LookupConv2d(3, 64, kernel_size=3, stride=1, padding=1, dictionary_size=100, sparsity=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 112 * 112, num_attributes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 64 * 112 * 112)
        x = self.fc(x)
        return x

# Load the dataset
dataset = load_dataset("lansinuote/gen.1.celeba")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a function to apply transformations
def transform_images(examples):
    examples['pixel_values'] = [transform(image.convert('RGB')) for image in examples['image']]
    return examples

# Apply the transformations to the dataset
print("Transforming images...")
dataset = dataset.map(transform_images, batched=True, remove_columns=['image'])

# Get the list of attribute columns
attribute_columns = [col for col in dataset['train'].column_names if col != 'pixel_values']
num_attributes = len(attribute_columns)

# Set the format of the dataset to PyTorch
dataset.set_format(type='torch', columns=['pixel_values'] + attribute_columns)

# Split the training set into train and validation
train_val_data = dataset['train']
train_size = int(0.8 * len(train_val_data))
val_size = len(train_val_data) - train_size
train_dataset, val_dataset = random_split(train_val_data, [train_size, val_size])

# Create DataLoaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize model, criterion, and optimizer
model = MyModel(num_attributes).cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
    for i, batch in pbar:
        inputs = batch['pixel_values'].cuda()
        labels = torch.stack([batch[col] for col in attribute_columns], dim=1).float().cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{running_loss / (i+1):.3f}'})

# Validation function
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            inputs = batch['pixel_values'].cuda()
            labels = torch.stack([batch[col] for col in attribute_columns], dim=1).float().cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{total_loss / (pbar.n+1):.3f}'})
    return total_loss / len(dataloader)

# Training loop with saving
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, criterion, optimizer, epoch)
    val_loss = validate(model, val_dataloader, criterion)
    print(f'Validation Loss: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, 'best_model.pth')
        print(f'Saved new best model with validation loss: {best_val_loss:.4f}')

# Inference function
def inference(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Inference")
        for batch in pbar:
            inputs = batch['pixel_values'].cuda()
            labels = torch.stack([batch[col] for col in attribute_columns], dim=1).float().cuda()
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_preds), torch.cat(all_labels)

# Load the best model and run inference
if os.path.exists('best_model.pth'):
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['loss']:.4f}")

print("Running inference...")
test_preds, test_labels = inference(model, val_dataloader)

# Calculate accuracy for each attribute
accuracy = (test_preds == test_labels).float().mean(dim=0)
for attr, acc in zip(attribute_columns, accuracy):
    print(f"{attr} accuracy: {acc:.4f}")

# Overall accuracy
overall_accuracy = accuracy.mean()
print(f"Overall accuracy: {overall_accuracy:.4f}")