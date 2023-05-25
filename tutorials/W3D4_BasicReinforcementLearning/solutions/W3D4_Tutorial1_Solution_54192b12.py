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
    # Pick an initial state randomly.
    self.current_state = np.random.choice(self.num_states)

  def step(self, action: int) -> Tuple[int, float]:
    """Take a step in MDP from self.current_state.

    Args:
      action: Action to take.

    Returns:
      Next state and reward received.
    """
    new_state = np.random.choice(self.num_states,
                                 p=self.P[self.current_state, action, :])
    return (new_state, self.R[self.current_state, action])

  def pickAction(self) -> int:
    """Pick the best action from the current state and Q-value estimates."""
    return np.argmax(self.Q[self.current_state, :])

  def maybeReset(self):
    """If current_state is goal, reset to a random state."""
    if self.current_state == self.goal_state:
      self.current_state = np.random.choice(self.num_states)

  def learnQ(self, alpha: float = 0.1, max_steps: int = 10_000):
    """Learn the Q-function by interacting with the environment.

    ** Assignment:** Write this function!
    Hint: Use the step(), pickAction(), and maybeReset() functions above.
    **Note: The way you initialize the Q-values is crucial here. Try first with
    an all-zeros initialization (as is currently coded below). If it doesn't
    work, try a different initialization.
    Hint: The maximum possible value (given the rewards are in [0, 1]) is
          1 / (1 - gamma).

    Args:
      alpha: Learning rate.
      max_steps: Maximum number of steps to take.
    """
    # Initialize Q-values optimistically.
    self.Q = np.ones((self.num_states, self.num_actions)) / (1 - self.gamma)
    num_steps = 0
    while num_steps < max_steps:
      a = self.pickAction()
      new_state, r = self.step(a)
      td = r + self.gamma * np.max(self.Q[new_state, :]) - self.Q[self.current_state, a]
      self.Q[self.current_state, a] += alpha * td
      self.current_state = new_state
      self.maybeReset()
      num_steps += 1
    self.V = np.max(self.Q, axis=-1)

  def plan(self):
    """Now planning is just doing an argmin over the Q-values!

    Note that this is a little different than standard Q-learning (where we do
    an argmax), since our Q-values currently store steps-to-go.
    """
    self.pi = np.argmax(self.Q, axis=-1)