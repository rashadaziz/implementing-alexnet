import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop, TenCrop, RandomHorizontalFlip, Lambda, ToTensor

training_dataset = load_dataset("ILSVRC/imagenet-1k", split="train")
validation_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")

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
    return train_loader, val_loader

def get_dataloaders_with_transforms(batch_size=32, num_workers=4):
    training_dataset.set_transform(train_transforms)
    validation_dataset.set_transform(test_transforms)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader

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