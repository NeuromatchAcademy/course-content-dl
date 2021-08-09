class Net4(nn.Module):
  def __init__(self, padding=0, stride=2):
    super(Net4, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3,
                            padding=padding)

    # first kernel - leading diagonal
    kernel_1 = torch.Tensor([[[ 1.,  1., -1., -1., -1.],
                              [ 1.,  1.,  1., -1., -1.],
                              [-1.,  1.,  1.,  1., -1.],
                              [-1., -1.,  1.,  1.,  1.],
                              [-1., -1., -1.,  1.,  1.]]])

    # second kernel - other diagonal
    kernel_2 = torch.Tensor([[[-1., -1., -1.,  1.,  1.],
                              [-1., -1.,  1.,  1.,  1.],
                              [-1.,  1.,  1.,  1., -1.],
                              [ 1.,  1.,  1., -1., -1.],
                              [ 1.,  1., -1., -1., -1.]]])

    # third kernel -checkerboard pattern
    kernel_3 = torch.Tensor([[[ 1.,  1., -1.,  1.,  1.],
                              [ 1.,  1.,  1.,  1.,  1.],
                              [-1.,  1.,  1.,  1., -1.],
                              [ 1.,  1.,  1.,  1.,  1.],
                              [ 1.,  1., -1.,  1.,  1.]]])


    # Stack all kernels in one tensor with (3, 1, 5, 5) dimensions
    multiple_kernels = torch.stack([kernel_1, kernel_2, kernel_3], dim=0)

    self.conv1.weight = torch.nn.Parameter(multiple_kernels)
    # Negative bias
    self.conv1.bias = torch.nn.Parameter(torch.Tensor([-4, -4, -12]))
    self.pool = nn.MaxPool2d(kernel_size=2, stride=stride)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool(x)  # pass through a max pool layer
    return x


# add event to airtable
atform.add_event('Coding Exercise 3.3: Implement MaxPooling')

## check if your implementation is correct
net4 = Net4().to(DEVICE)
check_pooling_net(net4, device=DEVICE)