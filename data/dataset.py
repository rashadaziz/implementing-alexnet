from datasets import load_dataset
from torch.utils.data import DataLoader

import inspect
import os
from pathlib import Path

DATA_PATH = Path(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe())))) / ".data"

training_dataset = load_dataset("ILSVRC/imagenet-1k", split="train")
validation_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")