from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomCrop, TenCrop, RandomHorizontalFlip, ToTensor

training_dataset = load_dataset("ILSVRC/imagenet-1k", split="train")
validation_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")

def get_dataloaders(batch_size=32, num_workers=4):
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def train_transforms(examples):
    transform = Compose(
        [
            Resize(256),
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
            TenCrop(224),
            ToTensor()
        ]
    )
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

training_dataset.set_transform(train_transforms)
validation_dataset.set_transform(test_transforms)

for example in training_dataset.select(range(5)):
    print(example['pixel_values'].shape)