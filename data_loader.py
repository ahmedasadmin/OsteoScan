import os 
from torch.utils.data import Dataset
from utils import * 
from PIL import Image 
import torchvision.transforms as T



class KneeXrayTransform:
    def __init__(self, size=(224, 224), normalize=True):
        transforms =[
            T.Resize((224, 224)),
            T.RandomRotation(10),  # realistic patient variation
            T.RandomAffine(degrees=0, translate=(0.05,0.05)),  # small translation
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor()
        ]


        if normalize:
            transforms.append(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))

        self.transform = T.Compose(transforms)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        return self.transform(image)



class KneeXrayDataset(Dataset):
    def __init__(self, image_paths, labels, reader, transform=None):
        self.image_paths = image_paths
        self.labels = labels 
        self.reader = reader
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]

        image = self.reader.read(img_path)

        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        
    
        
        if self.transform:
            image = self.transform(image)
        
        import torch
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label

