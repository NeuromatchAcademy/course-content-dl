class Net3(nn.Module):
  def __init__(self, kernel=None, padding=0, stride=2):
    super(Net3, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3,
                            padding=padding)

    # first kernel - leading diagonal
    kernel_1 = torch.Tensor([[[1, -1, -1],
                              [-1, 1, -1],
                              [-1, -1, 1]]])

    # second kernel -checkerboard pattern
    kernel_2 = torch.Tensor([[[1, -1, 1],
                              [-1, 1, -1],
                              [1, -1, 1]]])

    # third kernel - other diagonal
    kernel_3 = torch.Tensor([[[-1, -1, 1],
                            [-1, 1, -1],
                            [1, -1, -1]]])

    multiple_kernels = torch.stack([kernel_1, kernel_2, kernel_3], dim=0)

    self.conv1.weight = torch.nn.Parameter(multiple_kernels)
    self.conv1.bias = torch.nn.Parameter(torch.Tensor([0, 0, 0]))
    self.pool = nn.MaxPool2d(kernel_size=2, stride=stride)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool(x)  # pass through a max pool layer
    return x


## check if your implementation is correct
net = Net3().to(DEVICE)
check_pooling_net(net, device=DEVICE)