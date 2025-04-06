
import torch
from tqdm.auto import tqdm 
from typing import Dict, List, Tuple 

"""Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """

def train_step(
  model: torch.nn.Module,
  dataloader: torch.utils.data.DataLoader,
  loss_fn: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  device: torch.device
) -> Tuple[float, float]:

  model.train()

  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss 

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += ((y_pred_class == y).sum().item()/len(y_pred))

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)

  return train_loss, train_acc 

def test_step(
  model: torch.nn.Module,
  dataloader: torch.utils.data.DataLoader,
  loss_fn: torch.nn.Module,
  device: torch.device
) -> Tuple[float, float]:

  model.eval()

  test_loss, test_acc = 0, 0

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    test_pred = model(X)

    loss = loss_fn(test_pred, y)
    test_loss += loss 

    test_pred_labels = torch.argmax(test_pred, dim=1)
    test_acc += ((test_pred_labels == y).sum().item()/len(test_pred))

  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)

  return test_loss, test_acc

def train(
  model: torch.nn.Module,
  train_dataloader: torch.utils.data.DataLoader,
  test_dataloader: torch.utils.data.DataLoader,
  loss_fn: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  epochs: int,
  device: torch.device
) -> Dict[str, List]:

  results = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
  }

  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(
      model=model,
      dataloader=train_dataloader,
      loss_fn=loss_fn,
      optimizer=optimizer,
      device=device 
    )

    test_loss, test_acc = test_step(
      model=model,
      dataloader=test_dataloader,
      loss_fn=loss_fn,
      device=device
    )

    print(
      f"Epoch: {epoch+1} | ",
      f"train_loss: {train_loss:.4f} | ",
      f"train_acc: {train_acc:.4f} | ",
      f"test_loss: {test_loss:.4f} | ",
      f"test_acc: {test_acc:.4f} |"
    )

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results
