import torch
import torch.nn as nn
import os
from lookup_conv2d import LookupConv2d



class MyModel(nn.Module):
    def __init__(self, num_attributes):
        super(MyModel, self).__init__()
        self.conv1 = LookupConv2d(3, 64, kernel_size=3, stride=1, padding=1, dictionary_size=100, sparsity=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 112 * 112, num_attributes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 64 * 112 * 112)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


model = MyModel(num_attributes=40)

# Load the best model and run inference
if os.path.exists('best_model.pth'):
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['loss']:.4f}")



# Function to get predictions
def get_predictions(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        # Convert to boolean
        predictions = (output > 0.5).bool()
    return predictions

# List of CelebA attributes
celeba_attributes = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]


# Example usage:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Example input tensor
predictions = get_predictions(model, input_tensor)
for attr, pred in zip(celeba_attributes, predictions[0]):
    print(f"{attr}: {pred.item()}")