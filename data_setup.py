from torch.utils.data import DataLoader, Subset

"""Creates training and testing DataLoaders from given datasets.

  Args:
    train_data: A PyTorch Dataset object containing the training data.
    test_data: A PyTorch Dataset object containing the testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_data,
                             test_data,
                             batch_size=32)
  """

def create_dataloaders(
  train_data,
  test_data,
  batch_size: int,
):

  train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
  )

  test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
  )
  
  if isinstance(train_data, Subset):
    class_names = train_data.dataset.classes
  else:
    class_names = train_data.classes

  return train_dataloader, test_dataloader, class_names
