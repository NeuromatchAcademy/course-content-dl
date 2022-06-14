class PolicyGradientNet(nn.Module):
  """
  Defines Policy Gradient Network with the following attributes:
    Feed Forward Network with a single hidden layer
    width: 128 neurons
    dropout: 0.6
    Optimizer: Adam
    Learning Rate: 0.01
  """

  def __init__(self):
    """
    Initiate Policy Gradient Network with above mentioned parameters/hyperparameters

    Args:
      None

    Returns:
      Nothing
    """
    super(PolicyGradientNet, self).__init__()
    self.state_space = env.observation_space.shape[0]
    self.action_space = env.action_space.n
    # HINT: you can construct linear layers using nn.Linear(); what are the
    # sizes of the inputs and outputs of each of the layers? Also remember
    # that you need to use hidden_neurons (see hyperparameters section above).
    #   https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    self.l1 = nn.Linear(self.state_space, hidden_neurons, bias=False)
    self.l2 = nn.Linear(hidden_neurons, self.action_space, bias=False)

    self.gamma = gamma
    # Episode policy and past rewards
    self.past_policy = Variable(torch.Tensor())
    self.reward_episode = []
    # Overall reward and past loss
    self.past_reward = []
    self.past_loss = []

  def forward(self, x):
    model = torch.nn.Sequential(
        self.l1,
        nn.Dropout(p=dropout),
        nn.ReLU(),
        self.l2,
        nn.Softmax(dim=-1)
    )
    return model(x)