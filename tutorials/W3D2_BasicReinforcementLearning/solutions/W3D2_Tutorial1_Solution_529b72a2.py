class PolicyEvalAgent(acme.Actor):
  """
  New Acme agent that behaves according to uniform random policy.
  """

  def __init__(self, environment_spec, evaluated_policy,
               behaviour_policy=random_policy, step_size=0.1):
    """
    Initiates the agent

    Args:
      environment_spec: enum
        * actions: DiscreteArray(shape=(), dtype=int32, name=action, minimum=0, maximum=3, num_values=4)
        * observations: Array(shape=(9, 10, 3), dtype=dtype('float32'), name='observation_grid')
        * rewards: Array(shape=(), dtype=dtype('float32'), name='reward')
        * discounts: BoundedArray(shape=(), dtype=dtype('float32'), name='discount', minimum=0.0, maximum=1.0)
      evaluated_policy: f.__name__
        Policy on which agent is to be evaluated
      behavior policy: f.__name__
        Uniform random policy
      step_size: Float
        Size of step while choosing action

    Returns:
      Nothing
    """
    self._state = None
    # Get number of states and actions from the environment spec.
    self._number_of_states = environment_spec.observations.num_values
    self._number_of_actions = environment_spec.actions.num_values
    self._step_size = step_size
    self._behaviour_policy = behaviour_policy
    self._evaluated_policy = evaluated_policy
    # TODO Initialize the Q-values to be all zeros.
    # (Note: can also be random, but we use zeros here for reproducibility)
    # HINT: This is a table of state and action pairs, so needs to be a 2-D
    #   array. See the reference for how to create this in numpy:
    #   https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    self._q = np.zeros((self._number_of_states, self._number_of_actions))
    self._action = None
    self._next_state = None

  @property
  def q_values(self):
    # Return the Q values
    return self._q

  def select_action(self, observation):
    # Select an action
    return self._behaviour_policy(self._q[observation])

  def observe_first(self, timestep):
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
    s = self._state
    a = action
    r = next_timestep.reward
    g = next_timestep.discount
    next_s = next_timestep.observation

    # Compute TD-Error.
    self._action = a
    self._next_state = next_s
    # TODO Select the next action from the evaluation policy
    # HINT: Refer to step 4 of the algorithm above.
    next_a = self._evaluated_policy(self._q[next_s])
    self._td_error = r + g * self._q[next_s, next_a] - self._q[s, a]

  def update(self):
    """
    Perform update based on Q-table

    Args:
      None

    Returns:
      Nothing
    """
    # Updates
    s = self._state
    a = self._action
    # Q-value table update.
    self._q[s, a] += self._step_size * self._td_error
    # Update the state
    self._state = self._next_state


# Add event to airtable
atform.add_event('Coding Exercise 4.1 Policy Evaluation Agent')