class RandomAgent(acme.Actor):
  """
  Initiates Acme Random Agent.
  """

  def __init__(self, environment_spec):
    """
    Gets the number of available actions from the environment spec.

    Args:
      environment_spec: enum
        * actions: DiscreteArray(shape=(), dtype=int32, name=action, minimum=0, maximum=3, num_values=4)
        * observations: Array(shape=(9, 10, 3), dtype=dtype('float32'), name='observation_grid')
        * rewards: Array(shape=(), dtype=dtype('float32'), name='reward')
        * discounts: BoundedArray(shape=(), dtype=dtype('float32'), name='discount', minimum=0.0, maximum=1.0)

    Returns:
      Nothing
    """
    self._num_actions = environment_spec.actions.num_values

  def select_action(self, observation):
    """
    Selects an action uniformly at random.

    Args:
      observation_type: enum
        * ObservationType.STATE_INDEX: int32 index of agent occupied tile.
        * ObservationType.AGENT_ONEHOT: NxN float32 grid, with a 1 where the
          agent is and 0 elsewhere.
        * ObservationType.GRID: NxNx3 float32 grid of feature channels.
          First channel contains walls (1 if wall, 0 otherwise), second the
          agent position (1 if agent, 0 otherwise) and third goal position
          (1 if goal, 0 otherwise)
        * ObservationType.AGENT_GOAL_POS: float32 tuple with
          (agent_y, agent_x, goal_y, goal_x)

    Returns:
      action: List
        Selected action (uniformly at random).
    """
    # TODO return a random integer beween 0 and self._num_actions.
    # HINT: see the reference for how to sample a random integer in numpy:
    #   https://numpy.org/doc/1.16/reference/routines.random.html
    action = np.random.randint(self._num_actions)
    return action

  def observe_first(self, timestep):
    """ Does not record as the RandomAgent has no use for data. """
    pass

  def observe(self, action, next_timestep):
    """ Does not record as the RandomAgent has no use for data. """
    pass

  def update(self):
    """ Does not update as the RandomAgent does not learn from data. """
    pass


# Add event to airtable
atform.add_event('Coding Exercise 2.1: Random Agent')