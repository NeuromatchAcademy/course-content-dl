class MDPBase(object):
  """Creates a proper MDP from a GridWorld object."""

  def __init__(self, grid_world: GridWorldBase):
    """Constructs an MDP from a GridWorldBase object.

    Args:
      grid_world: GridWorld specification.
    """
    # Determine how many valid states there are and create empty matrices.
    self.grid_world = grid_world
    self.num_states = np.sum(grid_world.world_spec != '*')
    self.num_actions = len(ACTIONS)
    self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
    self.R = np.zeros((self.num_states, self.num_actions))
    self.pi = np.zeros(self.num_states, dtype=np.int32)

    # Build mapping between cell positions and state IDs (left→right, top→bottom).
    state_idx = 0
    self.cell_to_state = np.ones(grid_world.world_spec.shape, dtype=np.int32) * -1
    self.state_to_cell = {}
    for i, row in enumerate(grid_world.world_spec):
      for j, cell in enumerate(row):
        if cell == '*':
          continue
        if cell == 'g':
          self.goal_state = state_idx
        self.cell_to_state[i, j] = state_idx
        self.state_to_cell[state_idx] = (i, j)
        state_idx += 1

    # Populate P and R.
    for s in range(self.num_states):
      cell = self.state_to_cell[s]                     # (row, col) of state s
      neighbours = grid_world.get_neighbours(cell)      # dict: action → next (row,col)
      for a, action in enumerate(neighbours):
        nbr = neighbours[action]
        s2  = self.cell_to_state[nbr[0], nbr[1]]
        self.P[s, a, s2] = 1.0
        if s2 == self.goal_state:
          self.R[s, a] = 1.0

  def draw(self, include_policy: bool = False):
    for s in range(self.num_states):
      r, c = self.state_to_cell[s]
      self.grid_world.policy[r, c] = ACTIONS[self.pi[s]]
    self.grid_world.draw(include_policy)

  def plan(self):
    goal_queue = [self.goal_state]
    goals_done = set()
    while goal_queue:
      goal = goal_queue.pop(0)
      nbr_states, nbr_actions = np.where(self.P[:, :, goal] > 0.)
      goals_done.add(goal)
      for s, a in zip(nbr_states, nbr_actions):
        if s == goal:
          continue
        if s not in goals_done:
          self.pi[s] = a
          goal_queue.append(s)