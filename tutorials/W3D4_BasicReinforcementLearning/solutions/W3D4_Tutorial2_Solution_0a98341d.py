class QLearnerExplorer(QLearner):

  def __init__(self, grid_world: GridWorldBase, gamma: float = 0.99,
               epsilon: float = 0.1):
    """Constructs an MDP from a GridWorldBase object.

    States should be numbered from left-to-right and from top-to-bottom.

    Args:
      grid_world: GridWorld specification.
      gamma: Discount factor.
      epsilon: Exploration rate.
    """
    super().__init__(grid_world, gamma)
    self.epsilon = epsilon

  def pickAction(self) -> int:
    """With probability epsilon pick a random action; otherwise be greedy."""
    if np.random.rand() < self.epsilon:
      return np.random.choice(self.num_actions)   # explore: random action
    return np.argmax(self.Q[self.current_state, :])     # exploit: greedy action