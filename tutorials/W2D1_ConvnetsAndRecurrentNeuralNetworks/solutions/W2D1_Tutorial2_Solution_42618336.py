def get_augmentation_transforms():
  augmentation_transforms = [transforms.RandomRotation(10), transforms.RandomHorizontalFlip()]

  return augmentation_transforms


# add event to airtable
atform.add_event('Coding Exercise 2.2: How much does augmentation help?')

set_seed(SEED)
set_seed(SEED)
net3 = FMNIST_Net2().to(DEVICE)  # get the network

## Uncomment below to test your function
train_loader, validation_loader, test_loader = transforms_custom(SEED)
train_loss, train_acc, validation_loss, validation_acc = train(net3, DEVICE, train_loader, validation_loader, 20)
with plt.xkcd():
  plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc)