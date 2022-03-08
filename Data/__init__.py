from .aircraft import AircraftDataset
from .bird import BirdDataset
from .car import CarDataset
from .dog import DogDataset


def get_trainval_datasets(tag, resize, cropsize):
    if tag == 'aircraft':
        return AircraftDataset('train', resize, cropsize), AircraftDataset('val', resize, cropsize)
    elif tag == 'bird':
        return BirdDataset('train', resize, cropsize), BirdDataset('val', resize, cropsize)
    elif tag == 'car':
        return CarDataset('train', resize, cropsize), CarDataset('val', resize, cropsize)
    elif tag == 'dog':
        return DogDataset('train', resize, cropsize), DogDataset('val', resize, cropsize)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))
