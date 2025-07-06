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

def get_dataloaders(batch_size=32, num_workers=4):
    training_dataset.set_transform(basic_transform)
    validation_dataset.set_transform(basic_transform)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

def get_dataloaders_with_transforms(batch_size=32, num_workers=4):
    training_dataset.set_transform(train_transforms)
    validation_dataset.set_transform(test_transforms)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

def get_dataloaders_pca(sample_size=0.1, batch_size=256, num_workers=4):
    if sample_size > 1 or sample_size < 0:
        raise ValueError("Sample size must be between 0 and 1.")

    training_dataset.set_transform(pca_transform)
    validation_dataset.set_transform(pca_transform)

    pca_subsample = training_dataset.train_test_split(test_size=sample_size, stratify_by_column="label")
    pca_subsample_val = validation_dataset.train_test_split(test_size=sample_size, stratify_by_column="label")
    pca_subsample_test = test_dataset.train_test_split(test_size=sample_size, stratify_by_column="label")

    train_loader = DataLoader(pca_subsample["test"], batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(pca_subsample_val["test"], batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(pca_subsample_test["test"], batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

def train_transforms(examples):
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

def test_transforms(examples):
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

def basic_transform(examples):
    transform = Compose(
        [
            Resize(256),
            CenterCrop(256),
            ToTensor(),
        ]
    )
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

def pca_transform(examples):
    transform = Compose(
        [
            Resize(128),
            CenterCrop(128),
            ToTensor(),
        ]
    )
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples