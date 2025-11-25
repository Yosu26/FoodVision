import torch 
import torchvision 

from torch import nn 

def create_vit_model(num_classes:int=3,
                     seed:int=42):
  """Creates an ViTB16 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): ViTB16 feature extractor model. 
        transforms (torchvision.transforms): EffNetB2 image transforms.
  """
  # Create ViTB16 pretrained weights, transforms and model
  weights = torchvision.models.ViT_B_16_Weights.DEFAULT
  transforms = weights.transforms()
  model = torchvision.models.vit_b_16(weights=weights)

  # Freeze all layers in base model
  for param in model.parameters():
      param.requires_grad = False 

  # Change classifier head with random seed for reproducibility
  torch.manual_seed(seed)
  model.heads = nn.Sequential(
      nn.Linear(in_features=768, out_features=num_classes)
  )

  return model, transforms