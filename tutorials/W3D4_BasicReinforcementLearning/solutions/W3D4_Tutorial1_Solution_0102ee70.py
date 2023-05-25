class MDPBase(object):
  """This object creates a proper MDP from a GridWorld object."""

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
    # Create mapping from cell positions to state ID.
    state_idx = 0
    self.cell_to_state = np.ones(grid_world.world_spec.shape, dtype=np.int32) * -1  # Defaults to -1.
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
    # Assign transition probabilities and rewards accordingly.
    for s in range(self.num_states):
      neighbours = grid_world.get_neighbours(self.state_to_cell[s])
      for a, action in enumerate(neighbours):
        nbr = neighbours[action]
        s2 = self.cell_to_state[nbr[0], nbr[1]]
        self.P[s, a, s2] = 1.0  # Deterministic transitions
        if s2 == self.goal_state:
          self.R[s, a] = 1.0

  def draw(self, include_policy: bool = False):
    # First make sure we convert our MDP policy into the GridWorld policy.
    for s in range(self.num_states):
      r, c = self.state_to_cell[s]
      self.grid_world.policy[r, c] = ACTIONS[self.pi[s]]
    self.grid_world.draw(include_policy)

  def plan(self):
    """Define a planner
    """
    goal_queue = [self.goal_state]
    goals_done = set()
    while goal_queue:
      goal = goal_queue.pop(0)  # pop from front of list
      nbr_states, nbr_actions = np.where(self.P[:, :, goal] > 0.)
      goals_done.add(goal)
      for s, a in zip(nbr_states, nbr_actions):
        if s == goal:
          continue
        if s not in goals_done:
          self.pi[s] = a
          goal_queue.append(s)