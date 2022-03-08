import torch.utils.data as Data
import torchvision.transforms as transforms
from prefetch_generator import BackgroundGenerator

def get_transform(resize, cropsize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            # transforms.RandomResizedCrop(cropsize),
            transforms.Resize((resize, resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(cropsize),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(0.05, 0.05, 0.05),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class DataLoaderX(Data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())