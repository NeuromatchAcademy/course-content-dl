class Net3(nn.Module):
  def __init__(self, kernel=None, padding=0, stride=2):
    super(Net3, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3,
                            padding=padding)

    # first kernel
    kernel_1 = torch.unsqueeze(torch.Tensor([[1, -1, -1],
                                             [-1, 1, -1],
                                             [-1, -1, 1]]), 0)

    # second kernel
    kernel_2 = torch.unsqueeze(torch.Tensor([[1, -1, 1],
                                             [-1, 1, -1],
                                             [1, -1, 1]]), 0)

    # third kernel
    kernel_3 = torch.unsqueeze(torch.Tensor([[-1, -1, 1],
                                             [-1, 1, -1],
                                             [1, -1, -1]]), 0)

    multiple_kernels = torch.stack([kernel_1, kernel_2, kernel_3], dim=0)

    self.conv1.weight = torch.nn.Parameter(multiple_kernels)
    self.conv1.bias = torch.nn.Parameter(torch.zeros_like(self.conv1.bias))
    self.pool = nn.MaxPool2d(kernel_size=2, stride=stride)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool(x)  # pass through a max pool layer
    return x


net = Net3().to(device)
x_img = emnist_train[x_img_idx][0].unsqueeze(dim=0).to(device)
output_x = net(x_img)
output_x = output_x.squeeze(dim=0).detach().cpu().numpy()

o_img = emnist_train[o_img_idx][0].unsqueeze(dim=0).to(device)
output_o = net(o_img)
output_o = output_o.squeeze(dim=0).detach().cpu().numpy()