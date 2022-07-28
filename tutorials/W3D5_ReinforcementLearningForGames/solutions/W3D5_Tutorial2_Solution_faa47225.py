class OthelloNNet(nn.Module):
  """
  Instantiate Othello Neural Net with following configuration
  nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1) # Convolutional Layer 1
  nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1) # Convolutional Layer 2
  nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1) # Convolutional Layer 3
  nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1) # Convolutional Layer 4
  nn.BatchNorm2d(args.num_channels) X 4
  nn.Linear(args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024) # Fully-connected Layer 1
  nn.Linear(1024, 512) # Fully-connected Layer 2
  nn.Linear(512, self.action_size) # Fully-connected Layer 3
  nn.Linear(512, 1) # Fully-connected Layer 4
  """

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

    Returns:
      Nothing
    """
    self.board_x, self.board_y = game.getBoardSize()
    self.action_size = game.getActionSize()
    self.args = args

    super(OthelloNNet, self).__init__()
    self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1,
                           padding=1)
    self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
    self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

    self.bn1 = nn.BatchNorm2d(args.num_channels)
    self.bn2 = nn.BatchNorm2d(args.num_channels)
    self.bn3 = nn.BatchNorm2d(args.num_channels)
    self.bn4 = nn.BatchNorm2d(args.num_channels)

    self.fc1 = nn.Linear(args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024)
    self.fc_bn1 = nn.BatchNorm1d(1024)

    self.fc2 = nn.Linear(1024, 512)
    self.fc_bn2 = nn.BatchNorm1d(512)

    self.fc3 = nn.Linear(512, self.action_size)

    self.fc4 = nn.Linear(512, 1)

  def forward(self, s):
    """
    Controls forward pass of OthelloNNet

    Args:
      s: np.ndarray
        Array of size (batch_size x board_x x board_y)

    Returns:
      Probability distribution over actions at the current state and the value of the current state.
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

# Add event to airtable
atform.add_event('Coding Exercise 1.2: Implement the NN OthelloNNet for Othello')