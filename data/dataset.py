import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop, TenCrop, RandomHorizontalFlip, Lambda, ToTensor

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

training_dataset = load_dataset("ILSVRC/imagenet-1k", split="train")
validation_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")
test_dataset = load_dataset("ILSVRC/imagenet-1k", split="test")

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}

def _train_transforms(examples):
    transform = Compose(
        [
            Resize(256),
            CenterCrop(256),
            RandomCrop(224),
            RandomHorizontalFlip(),
            ToTensor()
        ]
    )
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

def _test_transforms(examples):
    transform = Compose(
        [
            Resize(256),
            CenterCrop(256),
            TenCrop(224),
            Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
        ]
    )
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

def _basic_transform(examples):
    transform = Compose(
        [
            Resize(256),
            CenterCrop(256),
            ToTensor(),
        ]
    )
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

def _pca_transform(examples):
    transform = Compose(
        [
            Resize(128),
            CenterCrop(128),
            ToTensor(),
        ]
    )
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

def get_dataloaders(sample_size=1, batch_size=32, num_workers=4, transform=_basic_transform):
    if sample_size > 1 or sample_size < 0:
        raise ValueError("Sample size must be between 0 and 1.")

    training_dataset.set_transform(transform)
    validation_dataset.set_transform(_test_transforms)
    test_dataset.set_transform(_test_transforms)

    if sample_size < 1:
        # Subsample the training dataset
        train_data = training_dataset.train_test_split(test_size=sample_size, stratify_by_column="label", seed=42)["test"]
    else:
        train_data = training_dataset

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

def get_dataloaders_training(sample_size=1, batch_size=32, num_workers=4):
    return get_dataloaders(sample_size, batch_size, num_workers, transform=_train_transforms)

def get_dataloaders_pca(sample_size=0.1, batch_size=256, num_workers=4):
    return get_dataloaders(sample_size, batch_size, num_workers, transform=_pca_transform)
