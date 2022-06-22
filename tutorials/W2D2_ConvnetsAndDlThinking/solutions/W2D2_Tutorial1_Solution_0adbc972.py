class FMNIST_Net2(nn.Module):
  """
  Neural Network instance
  """

  def __init__(self, num_classes):
    """
    Initialise parameters of FMNIST_Net2

    Args:
      num_classes: int
        Number of classes

    Returns:
      Nothing
    """
    super(FMNIST_Net2, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.25)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    """
    Forward pass of FMNIST_Net2

    Args:
      x: torch.tensor
        Input features

    Returns:
      x: torch.tensor
        Output after passing through FMNIST_Net2
    """
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    return x


set_seed(SEED)
## Uncomment below to check your code
net2 = FMNIST_Net2(num_classes=2).to(DEVICE)
train_loss, train_acc, validation_loss, validation_acc = train(net2, DEVICE, train_loader, validation_loader, 20)
print(f'Test accuracy is: {test(net2, DEVICE, test_loader)}')
with plt.xkcd():
  plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc)