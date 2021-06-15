def get_data_loaders(batch_size):
  # define the transform done only during training
  augmentation_transforms = [transforms.RandomRotation(10), transforms.RandomHorizontalFlip()]

  # define the transform done in training and testing (after augmentation)
  preprocessing_transforms = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

  # compose them together
  train_transform = transforms.Compose(augmentation_transforms + preprocessing_transforms)
  test_transform = transforms.Compose(preprocessing_transforms)

  # using pathlib to be compatible with all OS's
  data_path = pathlib.Path('.')/'afhq'

  # define the dataset objects (they can load one by one)
  img_train_dataset = ImageFolder(data_path/'train', transform=train_transform)
  img_test_dataset = ImageFolder(data_path/'val', transform=test_transform)

  # define the dataloader objects (they can load batch by batch)
  img_train_loader = DataLoader(img_train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=1, worker_init_fn=seed_worker)
  # num_workers can be set to higher if running on Colab Pro TPUs to speed up,
  # with more than one worker, it will do multithreading to queue batches
  img_test_loader = DataLoader(img_test_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=1)

  return img_train_loader, img_test_loader

### Uncomment below to test your function
batch_size = 64
img_train_loader, img_test_loader = get_data_loaders(batch_size)
# get some random training images
dataiter = iter(img_train_loader)
images, labels = dataiter.next()
# show images
imshow(make_grid(images, nrow=8))