class MDPPolicyIteration(MDPToGo):

  def __init__(self, grid_world: GridWorldBase, gamma: float = 0.99):
    """Constructs an MDP from a GridWorldBase object.

    States should be numbered from left-to-right and from top-to-bottom.

    Args:
      grid_world: GridWorld specification.
      gamma: Discount factor.
    """
    super().__init__(grid_world)
    self.gamma = gamma

  def findPi(self):
    """Find π policy.
    """
    self.Q = np.zeros((self.num_states, self.num_actions))
    self.pi = np.zeros(self.num_states, dtype=np.int32)
    num_iterations = 0
    new_pi = np.ones_like(self.pi)
    while np.any(new_pi != self.pi):
      new_pi = self.pi  # initialize to ones
      new_Q = np.zeros_like(self.Q)  # initialize to zeros
      next_V = np.array([mdpVi.Q[i, x] for i, x in enumerate(mdpVi.pi)])
      for a in range(self.num_actions):
        new_Q[:, a] = self.R[:, a] + self.gamma * np.matmul(self.P[:, a, :], next_V)
      self.Q = np.copy(new_Q)
      self.pi = np.argmax(self.Q, axis=-1)
      num_iterations += 1
    self.V = np.max(self.Q, axis=-1)
    print(f'Q and V found in {num_iterations} iterations.')

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