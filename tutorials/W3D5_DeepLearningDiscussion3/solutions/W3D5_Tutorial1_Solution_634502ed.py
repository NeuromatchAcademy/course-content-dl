class OthelloNNet(nn.Module):

  def __init__(self, game, args):
    """
    Initialise game parameters

    Args:
      game: OthelloGame instance
        Instance of the OthelloGame class above;
      args: dictionary
        Instantiates number of iterations and episodes, controls temperature threshold, queue length,
        arena, checkpointing, and neural network parameters:
        learning-rate: 0.001, dropout: 0.3, epochs: 10, batch_size: 64,
        num_channels: 512
    """
    self.board_x, self.board_y = game.getBoardSize()
    self.action_size = game.getActionSize()
    self.args = args

    super(OthelloNNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=args.num_channels,
                           kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=args.num_channels,
                           out_channels=args.num_channels, kernel_size=3,
                           stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=args.num_channels,
                           out_channels=args.num_channels, kernel_size=3,
                           stride=1)
    self.conv4 = nn.Conv2d(in_channels=args.num_channels,
                           out_channels=args.num_channels, kernel_size=3,
                           stride=1)

    self.bn1 = nn.BatchNorm2d(num_features=args.num_channels)
    self.bn2 = nn.BatchNorm2d(num_features=args.num_channels)
    self.bn3 = nn.BatchNorm2d(num_features=args.num_channels)
    self.bn4 = nn.BatchNorm2d(num_features=args.num_channels)

    self.fc1 = nn.Linear(in_features=args.num_channels * (self.board_x - 4) * (self.board_y - 4),
                         out_features=1024)
    self.fc_bn1 = nn.BatchNorm1d(num_features=1024)

    self.fc2 = nn.Linear(in_features=1024, out_features=512)
    self.fc_bn2 = nn.BatchNorm1d(num_features=512)

    self.fc3 = nn.Linear(in_features=512, out_features=self.action_size)

    self.fc4 = nn.Linear(in_features=512, out_features=1)

  def forward(self, s):
    """
    Controls forward pass of OthelloNNet

    Args:
      s: np.ndarray
        Array of size (batch_size x board_x x board_y)

    Returns:
      prob, v: tuple of torch.Tensor
        Probability distribution over actions at the current state and the value
        of the current state.
    """
    s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
    s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
    s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
    s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
    s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
    s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))

    s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
    s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

    pi = self.fc3(s)  # batch_size x action_size
    v = self.fc4(s)   # batch_size x 1
    # Returns probability distribution over actions at the current state and the value of the current state.
    return F.log_softmax(pi, dim=1), torch.tanh(v)