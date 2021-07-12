class PolicyEvalAgent(acme.Actor):

  def __init__(self, environment_spec, evaluated_policy,
               behaviour_policy=random_policy, step_size=0.1):

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
    # return the Q values
    return self._q

  def select_action(self, observation):
    # Select an action
    return self._behaviour_policy(self._q[observation])

  def observe_first(self, timestep):
    self._state = timestep.observation

  def observe(self, action, next_timestep):
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
    # Updates
    s = self._state
    a = self._action
    # Q-value table update.
    self._q[s, a] += self._step_size * self._td_error
    # Update the state
    self._state = self._next_state