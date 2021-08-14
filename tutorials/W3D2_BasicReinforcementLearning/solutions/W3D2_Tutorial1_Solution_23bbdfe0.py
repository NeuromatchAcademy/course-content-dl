class RandomAgent(acme.Actor):

  def __init__(self, environment_spec):
    """Gets the number of available actions from the environment spec."""
    self._num_actions = environment_spec.actions.num_values

  def select_action(self, observation):
    """Selects an action uniformly at random."""
    # TODO return a random integer beween 0 and self._num_actions.
    # HINT: see the reference for how to sample a random integer in numpy:
    #   https://numpy.org/doc/1.16/reference/routines.random.html
    action = np.random.randint(self._num_actions)
    return action

  def observe_first(self, timestep):
    """Does not record as the RandomAgent has no use for data."""
    pass

  def observe(self, action, next_timestep):
    """Does not record as the RandomAgent has no use for data."""
    pass

  def update(self):
    """Does not update as the RandomAgent does not learn from data."""
    pass


# add event to airtable
atform.add_event('Coding Exercise 2.1: Random Agent')