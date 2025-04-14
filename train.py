import argparse
import torch 

import data_setup, engine, model_builder, utils, get_data
from torchvision import transforms

def main():
  parser = argparse.ArgumentParser(description="Get some hyperparameters.")

  parser.add_argument(
    "--num_epochs",
    default=5,
    type=int,
    help="Number of epochs to train the model for."
  )

  parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="Number of samples per batch."
  )

  parser.add_argument(
    "--learning_rate",
    default=1e-3,
    type=float,
    help="Learning rate to use for the model"
  )
  
  parser.add_argument(
    "--use_food101",
    action="store_true",
    help="Use the Food101 dataset"
  )
  
  parser.set_defaults(use_food101=True)
  
  args = parser.parse_args()

  NUM_EPOCHS = args.num_epochs
  BATCH_SIZE = args.batch_size
  LEARNING_RATE = args.learning_rate

  print("[INFO] Training configuration:")
  for key, value in vars(args).items():
    print(f" {key}: {value}")

  device = "cuda" if torch.cuda.is_available() else "cpu"

  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

  # Get datasets
  if args.use_food101:
    train_data, test_data = get_data.get_food101_datasets()
  else:
    _, vit_transforms = model_builder.create_vit_model()

    train_data, test_data = get_data.get_imagefolder_datasets(
      train_dir=args.train_dir,
      test_dir=args.test_dir,
      transform=vit_transforms
    )
      
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_data,
    test_data,
    batch_size=BATCH_SIZE
  )
  
  vit, _ = model_builder.create_vit_model(num_classes=len(class_names), seed=42)
  
  loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
  optimizer = torch.optim.Adam(vit.parameters(), lr=LEARNING_RATE)

  engine.train(
    model=vit,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device
  )

  utils.save_model(
    model=vit,
    target_dir="models",
    model_name="pretrained_vit_model.pth"
  )

if __name__ == '__main__':
  main()
