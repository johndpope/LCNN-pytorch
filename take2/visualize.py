import torch
import matplotlib.pyplot as plt
from main import MyModel  # Import your model definition
import numpy as np
def visualize_lcnn(model, layer_name='conv1'):
    # Get the LookupConv2d layer
    layer = getattr(model, layer_name)
    
    # Extract dictionary, lookup indices, and coefficients
    dictionary = layer.dictionary.detach().cpu().numpy()
    lookup_indices = layer.lookup_indices.cpu().numpy()
    lookup_coefficients = layer.lookup_coefficients.detach().cpu().numpy()
    
    # Get dimensions
    dict_size, in_channels, kernel_size, _ = dictionary.shape
    out_channels, sparsity = lookup_indices.shape
    
    # Visualize dictionary
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    fig.suptitle("Dictionary Elements", fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < dict_size:
            # Combine channels for visualization
            dict_elem = np.sum(dictionary[i], axis=0)
            ax.imshow(dict_elem, cmap='viridis')
            ax.set_title(f"Dict {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Visualize constructed filters
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))
    fig.suptitle("Constructed Filters", fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < out_channels:
            # Construct filter
            filter = np.zeros((in_channels, kernel_size, kernel_size))
            for s in range(sparsity):
                dict_idx = lookup_indices[i, s]
                coeff = lookup_coefficients[i, s]
                filter += coeff * dictionary[dict_idx]
            # Combine channels for visualization
            filter = np.sum(filter, axis=0)
            ax.imshow(filter, cmap='viridis')
            ax.set_title(f"Filter {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Visualize filter construction
    fig, axes = plt.subplots(8, sparsity + 1, figsize=(20, 30))
    fig.suptitle("Filter Construction", fontsize=16)
    for i in range(8):  # Visualize 8 filters
        filter = np.zeros((in_channels, kernel_size, kernel_size))
        for s in range(sparsity):
            dict_idx = lookup_indices[i, s]
            coeff = lookup_coefficients[i, s]
            component = coeff * dictionary[dict_idx]
            filter += component
            
            # Visualize component
            axes[i, s].imshow(np.sum(component, axis=0), cmap='viridis')
            axes[i, s].set_title(f"Dict {dict_idx}\nCoeff {coeff:.2f}")
            axes[i, s].axis('off')
        
        # Visualize final filter
        axes[i, -1].imshow(np.sum(filter, axis=0), cmap='viridis')
        axes[i, -1].set_title("Final Filter")
        axes[i, -1].axis('off')
    
    plt.tight_layout()
    plt.show()

    # Print additional information
    print(f"Layer: {layer_name}")
    print(f"Dictionary shape: {dictionary.shape}")
    print(f"Lookup indices shape: {lookup_indices.shape}")
    print(f"Lookup coefficients shape: {lookup_coefficients.shape}")
    print(f"Number of output channels: {out_channels}")
    print(f"Sparsity: {sparsity}")
    print(f"Kernel size: {kernel_size}")
    
    # Print a few lookup indices and coefficients
    print("\nSample Lookup Indices and Coefficients:")
    for i in range(5):  # Print for first 5 filters
        print(f"Filter {i}:")
        print(f"  Indices: {lookup_indices[i]}")
        print(f"  Coefficients: {lookup_coefficients[i]}")


# Load your trained model
model = MyModel(num_attributes=40)  # Adjust as needed
model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])

# Visualize the LCNN
visualize_lcnn(model, 'conv1')  # Assuming 'conv1' is the name of your LookupConv2d layer