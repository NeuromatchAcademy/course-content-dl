class PolicyGradientNet(nn.Module):
  def __init__(self):
    super(PolicyGradientNet, self).__init__()
    self.state_space = env.observation_space.shape[0]
    self.action_space = env.action_space.n
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