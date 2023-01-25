def get_data_loaders(batch_size, seed):
  """
  Helper function to get data loaders

  Args:
    batch_size: int
      Batch size
    seed: int
      A non-negative integer that defines the random state.

  Returns:
    img_train_loader: torch.utils.data type
      Combines the train dataset and sampler, and provides an iterable over the given dataset.
    img_test_loader: torch.utils.data type
      Combines the test dataset and sampler, and provides an iterable over the given dataset.
  """
  # Define the transform done only during training
  augmentation_transforms = [transforms.RandomRotation(10), transforms.RandomHorizontalFlip()]

  # Define the transform done in training and testing (after augmentation)
  mean = (0.5, 0.5, 0.5) # defined sequence of means per channel
  std = (0.5, 0.5, 0.5) # defined sequence of std deviations per channel
  # Note that the transform should normalize each channel: output[channel] = (input[channel] - mean[channel]) / std[channel]
  preprocessing_transforms = [transforms.ToTensor(), transforms.Normalize(mean, std)]

  # Compose them together
  train_transform = transforms.Compose(augmentation_transforms + preprocessing_transforms)
  test_transform = transforms.Compose(preprocessing_transforms)

  # Using pathlib to be compatible with all OS's
  data_path = pathlib.Path('.')/'afhq'

  # Define the dataset objects (they can load one by one)
  img_train_dataset = ImageFolder(data_path/'train', transform=train_transform)
  img_test_dataset = ImageFolder(data_path/'val', transform=test_transform)

  g_seed = torch.Generator()
  g_seed.manual_seed(seed)
  # Define the dataloader objects (they can load batch by batch)
  img_train_loader = DataLoader(img_train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=g_seed)
  # num_workers can be set to higher if running on Colab Pro TPUs to speed up,
  # with more than one worker, it will do multithreading to queue batches
  img_test_loader = DataLoader(img_test_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=1,
                               worker_init_fn=seed_worker,
                               generator=g_seed)

  return img_train_loader, img_test_loader


# Add event to airtable
atform.add_event('Coding Exercise 2: Dataloader on a real-world dataset')


batch_size = 64
set_seed(seed=SEED)
## Uncomment below to test your function
img_train_loader, img_test_loader = get_data_loaders(batch_size, SEED)
## get some random training images
dataiter = iter(img_train_loader)
images, labels = next(dataiter)
## show images
imshow(make_grid(images, nrow=8))