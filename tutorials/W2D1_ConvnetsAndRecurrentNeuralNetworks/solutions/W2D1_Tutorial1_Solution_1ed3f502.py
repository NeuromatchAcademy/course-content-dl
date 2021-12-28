def get_augmentation_transforms():
  augmentation_transforms = [transforms.RandomRotation(10), transforms.RandomHorizontalFlip()]

  return augmentation_transforms


set_seed(SEED)
net3 = FMNIST_Net2(num_classes=2).to(DEVICE)  # get the network

## Uncomment below to test your function
train_loader, validation_loader, test_loader = transforms_custom(binary=True, seed=SEED)
train_loss, train_acc, validation_loss, validation_acc = train(net3, DEVICE, train_loader, validation_loader, 20)
print(f'Test accuracy is: {test(net3, DEVICE, test_loader)}')
with plt.xkcd():
  plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc)