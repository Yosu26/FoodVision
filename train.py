
import os 
import argparse
import torch 
from torchvision import transforms

import data_setup, engine, model_builder, utils

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="Get some hyperparameters.")

  parser.add_argument(
    "--num_epochs",
    default=10,
    type=int,
    help="Number of epochs to train the model for."
  )

  parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="Number of samples per batch."
  )

  parser.add_argument(
    "--hidden_units",
    default=10,
    type=int,
    help="Number of hidden units per hidden layer."
  )

  parser.add_argument(
    "--learning_rate",
    default=0.001,
    type=float,
    help="Learning rate to use for the model"
  )

  parser.add_argument(
    "--train_dir",
    default="data/pizza_steak_sushi/train",
    type=str,
    help="Directory file path to training data in standard image classification format."
  )

  parser.add_argument(
    "--test_dir",
    default="data/pizza_steak_sushi/test",
    type=str,
    help="Directory file path to testing data in standard image classification format."
  )

  args = parser.parse_args()

  NUM_EPOCHS = args.num_epochs
  BATCH_SIZE = args.batch_size
  HIDDEN_UNITS = args.hidden_units
  LEARNING_RATE = args.learning_rate

  print("[INFO] Training configuration:")
  for key, value in vars(args).items():
    print(f" {key}: {value}")

  train_dir = args.train_dir
  test_dir = args.test_dir 

  print(f"[INFO] Training data file: {train_dir}")
  print(f"[INFO] Testing data file: {test_dir}")

  device = "cuda" if torch.cuda.is_available() else "cpu"

  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

  data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
  ])

  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
  )

  model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
  ).to(device)

  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device
  )

  utils.save_model(
    model=model,
    target_dir="models",
    model_name="tiny_vgg_model.pth"
  )
