from torchvision import datasets, transforms
from model_builder import create_vit_model
from utils import split_dataset
from pathlib import Path

def get_food101_datasets(data_dir: str="data"):
  # Setup data directory
  data_dir = Path(data_dir)

  # Create Food101 training data transform
  _, vit_transforms = create_vit_model()

  train_transforms = transforms.Compose([
    transforms.TrivialAugmentWide(),
    vit_transforms
  ])

  # Get training data
  train = datasets.Food101(root=data_dir,
                                split="train",
                                transform=train_transforms,
                                download=True)

  test = datasets.Food101(root=data_dir,
                              split="test",
                              transform=vit_transforms,
                              download=True)
  
  train_data, _ = split_dataset(dataset=train,
                                      split_size=0.2)

  test_data, _ = split_dataset(dataset=test,
                                      split_size=0.2)
  
  return train_data, test_data

def get_imagefolder_datasets(
  train_dir: str,
  test_dir: str,
  transform: transforms.Compose
):

  # Get training data
  train_data = datasets.ImageFolder(train_dir,
                                    transform=transform)
  
  test_data = datasets.ImageFolder(test_dir,
                                   transform=transform)
  
  return train_data, test_data