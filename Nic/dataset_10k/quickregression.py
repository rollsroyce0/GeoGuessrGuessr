import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_path = os.path.dirname(os.path.realpath(__file__))
csv_file_path = os.path.join(dir_path, 'coords.csv')

class GeoDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file, names=['x-coordinate', 'y-coordinate'])
        self.image_paths = [os.path.join(dir_path, 'dataset', f'{i}.png') for i in range(0, len(self.dataframe))]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        coordinates = self.dataframe.iloc[idx].values
        return image, coordinates

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.base_model(x)

# Create the dataset
dataset = GeoDataset(csv_file_path)
image, coordinates = dataset[0]
print("Coordinates:", coordinates)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Net()
model = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):  # loop over the dataset multiple times
    print(epoch)
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.float(), labels.float()  # Convert to FloatTensor

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')