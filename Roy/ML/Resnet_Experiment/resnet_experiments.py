import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rich.progress import track
from torch.optim.lr_scheduler import ReduceLROnPlateau

def resnet_layer_values_counter(model):
    layer_values = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_name = name.split('.')[0]
            if layer_name not in layer_values:
                layer_values[layer_name] = 0
            layer_values[layer_name] += param.numel()
    return layer_values

# Example usage:
resnet18 = models.resnet152(pretrained=True)
layer_values = resnet_layer_values_counter(resnet18)
for layer, value in layer_values.items():
    print(f"Layer: {layer}, Number of parameters: {value}")
    
resnet18.eval()  # Set the model to evaluation mode
dummy_input = torch.randn(1, 3, 224, 224)  # Example input tensor
output = resnet18(dummy_input)
#print(output)
print(torch.Tensor.size(output))  # Print the output shape

#remove the last layer
modules = list(resnet18.children())[:-1]
resnet18_modified = nn.Sequential(*modules)
resnet18_modified.eval()
output_modified = resnet18_modified(dummy_input)
print(torch.Tensor.size(output_modified))  # Print the output shape after removing last layer

#remove the last two layers
modules = list(resnet18.children())[:-2]
resnet18_modified_2 = nn.Sequential(*modules)
resnet18_modified_2.eval()
output_modified_2 = resnet18_modified_2(dummy_input)
output_modified_2 = output_modified_2.view(output_modified_2.size(0), -1)  # Flatten the output
print(torch.Tensor.size(output_modified_2))  # Print the output shape after removing last two layers