class PolicyEvalAgent(acme.Actor):

  def __init__(self, number_of_states, number_of_actions, evaluated_policy,
               behaviour_policy=random_policy, step_size=0.1):

    self._state = None
    self._number_of_states = number_of_states
    self._number_of_actions = number_of_actions
    self._step_size = step_size
    self._behaviour_policy = behaviour_policy
    self._evaluated_policy = evaluated_policy
    # (this is a table of state and action pairs)
    # Note: this can be random, but the code was tested w/ zero-initialization
    self._q = np.zeros((number_of_states, number_of_actions))
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
    self._td_error = r + g * self._q[next_s, next_a] - self._q[s, a]

  def update(self):
    # Updates
    s = self._state
    a = self._action
    # Q-value table update.
    self._q[s, a] += self._step_size * self._td_error
    # Update the state
    self._state = self._next_state