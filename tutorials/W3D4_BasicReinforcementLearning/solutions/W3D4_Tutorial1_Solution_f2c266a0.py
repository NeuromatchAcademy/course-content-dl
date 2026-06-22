class MDPToGo(MDPBase):

  def __init__(self, grid_world: GridWorldBase):
    """Constructs an MDP from a GridWorldBase object.

    States should be numbered from left-to-right and from top-to-bottom.

    Args:
      grid_world: GridWorld specification.
    """
    super().__init__(grid_world)
    self.Q = np.zeros((self.num_states, self.num_actions))

  def computeQ(self):
    """Store discounted steps-to-go in an SxA matrix called Q.

    This matrix will then be used to extract the optimal policy.
    """
    goal_queue = [(self.goal_state, 0)]  # (state, steps taken so far)
    goals_done = set()
    while goal_queue:
      goal, steps_to_go = goal_queue.pop(0)
      steps_to_go += 1
      nbr_states, nbr_actions = np.where(self.P[:, :, goal] > 0.)
      goals_done.add(goal)

      for s, a in zip(nbr_states, nbr_actions):
        if goal == self.goal_state and s == self.goal_state:
          self.Q[s, a] = 0        # action at goal that stays at goal
        elif s == goal:
          self.Q[s, a] = np.inf        # If (s, a) leads to itself then we have an infinite loop
        else:
          self.Q[s, a] = steps_to_go        # normal: steps_to_go steps to goal
        if s not in goals_done:
          goal_queue.append((s, steps_to_go))

  def plan(self):
    """Now planning is just doing an argmin over the Q-values!

    Note that this is a little different than standard Q-learning (where we do
    an argmax), since our Q-values currently store steps-to-go.
    """
    self.pi = np.argmin(self.Q, axis=-1)