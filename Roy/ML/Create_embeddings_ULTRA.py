import os
import torch
import numpy as np # Import joblib for saving and loading models
from PIL import Image
from rich.progress import track
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import warnings


warnings.filterwarnings("ignore")

# Function to extract coordinates from image path
def extract_coordinates(image_path):
    lat = float(image_path.split('_')[0].replace('D:/GeoGuessrGuessr/geoguesst\\', ''))
    lon = float(image_path.split('_')[1])
    return lat, lon

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the location of the images
location = "D:/GeoGuessrGuessr/geoguesst"

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class AugmentedStreetviewDataset(Dataset):
    def __init__(self, image_paths, coordinates):
        self.image_paths = image_paths
        self.coordinates = coordinates
        
        # Define 10 different transform pipelines without horizontal flip.
        # Two of these are dedicated to producing only high-res or only low-res images.
        self.transform_pipelines = [
            # 1. Only high-res: just resize and normalize.
            transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            # 2. Only low-res: just resize and normalize.
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            # 3. High-res with color jitter.
            transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            # 4. High-res with random perspective.
            transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            # 5. Low-res with random crop.
            transforms.Compose([
                transforms.Resize((300, 300)),  # oversize for crop
                transforms.RandomCrop((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            # 6. Low-res with Gaussian blur.
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            # 7. High-res with random rotation.
            transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            # 8. Low-res with perspective and color jitter.
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            # 9. High-res with random grayscale.
            transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            # 10. High-res with random crop.
            transforms.Compose([
                transforms.Resize((1100, 1100)),  # oversize for crop
                transforms.RandomCrop((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        ]

    def __len__(self):
        # Each image will produce 10 augmented versions.
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        # Apply all 10 transformation pipelines.
        augmented_images = [pipeline(image) for pipeline in self.transform_pipelines]
        #print(f"Augmented images shape: {[img.shape for img in augmented_images]}")
        return augmented_images, self.coordinates[idx]



# Custom Dataset
class DualResStreetviewDataset(Dataset):
    def __init__(self, image_paths):
        
        self.image_paths = image_paths
        self.coordinates = coordinates

        self.transform_large = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_small = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return 2 * len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx // 2]
        image = Image.open(path).convert("RGB")
        if idx % 2 == 0:
            return self.transform_large(image)
        else:
            return self.transform_small(image)

# Custom Model
class GeoEmbeddingModel(nn.Module):
    def __init__(self):
        super(GeoEmbeddingModel, self).__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove final classification layer

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x
print("---------------------------------------------------------------------------------------------------------------------------------------------")
# Load or generate embeddings
if os.path.exists('Roy/ML/embeddings.npy') and os.path.exists('Roy/ML/image_paths.npy'):
    print("Loading existing embeddings...")
    embeddings = np.load('Roy/ML/embeddings.npy').astype(np.float32)  # Ensure embeddings are float64
    image_paths = np.load('Roy/ML/image_paths.npy')
else:
    print("Generating new embeddings...")
    # Load image paths and extract coordinates
    image_paths = [os.path.join(location, img_file) for img_file in os.listdir(location)]
    coordinates = np.array([extract_coordinates(path) for path in image_paths])

    print(f"Number of images: {len(image_paths)}")
    
    # Initialize the custom model
    model = GeoEmbeddingModel().to(device)
    
    print("Loading model...")
    
    # Dataset and DataLoader
    dataset = AugmentedStreetviewDataset(image_paths=image_paths, coordinates=coordinates)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    print("DataLoader created.")

    # Load the pre-trained model if available
    if os.path.exists('Roy/ML/Saved_Models/geo_embedding_model.pth'):
        model.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model.pth'))

        print("Model loaded successfully.")
    else:
        print("No pre-trained model found. Training from scratch.")

    # Generate embeddings
    print("Generating custom embeddings...")
    embeddings = []
    embedding_paths = []

    model.eval()
    with torch.no_grad():
        for batch_aug_images, batch_coords in track(dataloader, description="Processing images..."):
            for i, aug_images in enumerate(batch_aug_images):  # i is the index in the batch
                image_path = dataset.image_paths[i]  # original path
                for aug_img in aug_images:  # iterate over the 10 augmentations
                    aug_img = aug_img.to(device)
                    output = model(aug_img.unsqueeze(0))
                    embeddings.append(output.cpu().numpy().astype(np.float32))
                    embedding_paths.append(image_path)  # store path once per augmentation


    embeddings = np.vstack(embeddings)
    embedding_paths = np.array(embedding_paths)

    # Save
    np.save('Roy/ML/embeddings.npy', embeddings)
    np.save('Roy/ML/image_paths.npy', embedding_paths)

    
    # save the model
    torch.save(model.state_dict(), 'Roy/ML/Saved_Models/geo_embedding_model.pth')

print(f"Number of images: {len(image_paths)}")
print(f"Embeddings shape: {embeddings.shape}")

