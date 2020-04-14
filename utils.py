import torchvision
from vision.references.detection import utils
import vision.references.detection.transforms as T

def get_transform(train):
    # data augmentation
    transforms = []
    transforms.append(T.ToTensor())

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))