class QLearner(MDPValueIteration):

  def __init__(self, grid_world: GridWorldBase, gamma: float = 0.99):
    """Constructs an MDP from a GridWorldBase object.

    States should be numbered from left-to-right and from top-to-bottom.

    Args:
      grid_world: GridWorld specification.
      gamma: Discount factor.
    """
    super().__init__(grid_world, gamma)
    self.Q = np.zeros((self.num_states, self.num_actions))
    self.current_state = np.random.choice(self.num_states)

  def step(self, action: int) -> Tuple[int, float]:
    """Take one step: returns (next_state, reward)."""
    new_state = np.random.choice(self.num_states,
                                 p=self.P[self.current_state, action, :])
    return (new_state, self.R[self.current_state, action])

  def pickAction(self) -> int:
    """Greedy action from current state."""
    return np.argmax(self.Q[self.current_state, :])

  def maybeReset(self):
    """Reset to a random state when the goal is reached."""
    if self.current_state == self.goal_state:
      self.current_state = np.random.choice(self.num_states)

  def learnQ(self, alpha: float = 0.1, max_steps: int = 10_000):
    """Learn Q by interacting with the environment.

    Args:
      alpha: Learning rate.
      max_steps: Total number of environment steps.
    """
    # Initialise Q optimistically so the agent is forced to explore.
    # The maximum possible discounted return is 1 / (1 - gamma).
    self.Q = np.ones((self.num_states, self.num_actions)) / (1 - self.gamma)
    num_steps = 0
    while num_steps < max_steps:
      a = self.pickAction()                          # pick a greedy action
      new_state, r = self.step(a)               # take the step
      td = r + self.gamma * np.max(self.Q[new_state, :]) - self.Q[self.current_state, a]                         # TD error: r + γ·max(Q[s',:]) - Q[s,a]
      self.Q[self.current_state, a] += alpha * td   # α · TD error
      self.current_state = new_state
      self.maybeReset()
      num_steps += 1
    self.V = np.max(self.Q, axis=-1)

  def plan(self):
    self.pi = np.argmax(self.Q, axis=-1)
    self.V  = np.max(self.Q, axis=-1)