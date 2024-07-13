import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
import os

from lookup_conv2d import LookupConv2d

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = LookupConv2d(3, 64, kernel_size=3, stride=1, padding=1, dictionary_size=100, sparsity=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 112 * 112, 40)  # Adjusted for 224x224 input and 40 attributes

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
dataset = dataset.map(transform_images, batched=True, remove_columns=['image'])

# Set the format of the dataset to PyTorch
dataset.set_format(type='torch', columns=['pixel_values', 'attributes'])

# Create DataLoaders
batch_size = 32
train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset['validation'], batch_size=batch_size)
test_dataloader = DataLoader(dataset['test'], batch_size=batch_size)

# Initialize model, criterion, and optimizer
model = MyModel().cuda()
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        inputs = batch['pixel_values'].cuda()
        labels = batch['attributes'].float().cuda()  # Convert to float for BCE loss
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

# Validation function
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].cuda()
            labels = batch['attributes'].float().cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop with saving
num_epochs = 10
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train(model, train_dataloader, criterion, optimizer, epoch)
    val_loss = validate(model, val_dataloader, criterion)
    print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')
    
    # Save the model if it's the best so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, 'best_model.pth')
        print(f'Saved new best model with validation loss: {best_val_loss:.4f}')

# Inference code
def inference(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].cuda()
            labels = batch['attributes'].cuda()
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()  # Convert to binary predictions
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_preds), torch.cat(all_labels)

# Load the best model
if os.path.exists('best_model.pth'):
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['loss']:.4f}")

# Run inference on test set
test_preds, test_labels = inference(model, test_dataloader)

# Calculate accuracy for each attribute
accuracy = (test_preds == test_labels).float().mean(dim=0)
for i, acc in enumerate(accuracy):
    print(f"Attribute {i} accuracy: {acc:.4f}")

# Overall accuracy
overall_accuracy = accuracy.mean()
print(f"Overall accuracy: {overall_accuracy:.4f}")