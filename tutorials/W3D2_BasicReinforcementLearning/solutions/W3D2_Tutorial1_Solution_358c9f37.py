QValues = np.ndarray
Action = int

# A value-based policy takes the Q-values at a state and returns an action.
ValueBasedPolicy = Callable[[QValues], Action]


class QLearningAgent(acme.Actor):
  """
  Build QLearning Agent
  """

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               behaviour_policy: ValueBasedPolicy,
               step_size: float = 0.1):
    """
    Initiates QLearning Agent

    Args:
      environment_spec: enum
        * actions: DiscreteArray(shape=(), dtype=int32, name=action, minimum=0, maximum=3, num_values=4)
        * observations: Array(shape=(9, 10, 3), dtype=dtype('float32'), name='observation_grid')
        * rewards: Array(shape=(), dtype=dtype('float32'), name='reward')
        * discounts: BoundedArray(shape=(), dtype=dtype('float32'), name='discount', minimum=0.0, maximum=1.0)
      behaviour_policy: f.__name__
        Policy based on which agent behaves
      step_size: Float
        Size of step while choosing action [default: 0.1]

    Returns:
      Nothing
    """
    # Get number of states and actions from the environment spec.
    self._num_states = environment_spec.observations.num_values
    self._num_actions = environment_spec.actions.num_values

    # Create the table of Q-values, all initialized at zero.
    self._q = np.zeros((self._num_states, self._num_actions))

    # Store algorithm hyper-parameters.
    self._step_size = step_size

    # Store behavior policy.
    self._behaviour_policy = behaviour_policy

    # Containers you may find useful.
    self._state = None
    self._action = None
    self._next_state = None

  @property
  def q_values(self):
    return self._q

  def select_action(self, observation):
    return self._behaviour_policy(self._q[observation])

  def observe_first(self, timestep):
    # Set current state.
    self._state = timestep.observation

  def observe(self, action, next_timestep):
    """
    Function to compute TD Error

    Args:
      action: Integer
        Selected action based on Q value
      next_timestep: dm_env._environment.TimeStep
        Advances timestep

    Returns:
      Nothing
    """
    # Unpacking the timestep to lighten notation.
    s = self._state
    a = action
    r = next_timestep.reward
    g = next_timestep.discount
    next_s = next_timestep.observation

    # Compute the TD error.
    self._action = a
    self._next_state = next_s
    # TODO complete the line below to compute the temporal difference error
    # HINT: This is very similar to what we did for SARSA, except keep in mind
    # that we're now taking a max over the q-values (see lecture footnotes above).
    # You will find the function np.max() useful.
    self._td_error = r + g * np.max(self._q[next_s]) - self._q[s, a]

  def update(self):
    """
    Perform update based on Q-table

    Args:
      None

    Returns:
      Nothing
    """
    # Optional unpacking to lighten notation.
    s = self._state
    a = self._action
    # Update the Q-value table value at (s, a).
    # TODO: Update the Q-value table value at (s, a).
    # HINT: see step 6 in the pseudocode above, remember that alpha = step_size!
    self._q[s, a] += self._step_size * self._td_error
    # Update the current state.
    self._state = self._next_state


# Add event to airtable
atform.add_event('Coding Exercise 5.3: Implement Q-Learning')