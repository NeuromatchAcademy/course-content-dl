class MDPValueIteration(MDPToGo):

  def __init__(self, grid_world: GridWorldBase, gamma: float = 0.99):
    """Constructs an MDP from a GridWorldBase object.

    States should be numbered from left-to-right and from top-to-bottom.

    Args:
      grid_world: GridWorld specification.
      gamma: Discount factor.
    """
    super().__init__(grid_world)
    self.gamma = gamma

  def computeQ(self, error_tolerance : float = 1e-5):
    """Compute Q and V vectors via value iteration.

    Args:
      error_tolerance: How much error we tolerate between successive Q updates.
    """
    self.Q = np.zeros((self.num_states, self.num_actions))
    num_iterations = 0
    error = np.inf
    while error > error_tolerance:
      new_Q = np.zeros_like(self.Q)
      max_next_Q = np.max(self.Q, axis=-1)
      for a in range(self.num_actions):
        new_Q[:, a] = self.R[:, a] + self.gamma * (self.P[:, a, :] @ max_next_Q)
      error = np.max(abs(new_Q - self.Q))
      self.Q = np.copy(new_Q)
      num_iterations += 1
    self.V = np.max(self.Q, axis=-1)
    print(f'Q and V found in {num_iterations} iterations with an error tolerance of {error_tolerance}.')

  def plan(self):
    """Now planning is just doing an argmax over the Q-values!
    """
    self.pi = np.argmax(self.Q, axis=-1)

  def _draw_v(self):
    """Draw the V values."""
    min_v = np.min(self.V)
    max_v = np.max(self.V)
    wall_v = 2 * min_v - max_v  # Creating a smaller value for walls.
    grid_values = np.ones_like(self.grid_world.world_spec, dtype=np.int32) * wall_v
    # Fill in the V values in grid cells.
    for s in range(self.num_states):
      cell = self.state_to_cell[s]
      grid_values[cell[0], cell[1]] = self.V[s]

    fig, ax = plt.subplots()
    ax.grid(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    grid = ax.matshow(grid_values)
    grid.set_clim(wall_v, max_v)
    fig.colorbar(grid)

  def draw(self, draw_mode: str = 'grid'):
    """Draw the GridWorld according to specified mode.

    Args:
      draw_mode: Specification of what mode to draw. Supported options:
                 'grid': Draw the base GridWorld.
                 'policy': Display the policy.
                 'values': Display the values for each state.
    """
    # First make sure we convert our MDP policy into the GridWorld policy.
    if draw_mode == 'values':
      self._draw_v()
    else:
      super().draw(draw_mode == 'policy')