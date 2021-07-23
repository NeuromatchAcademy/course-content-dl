class SarsaAgent(acme.Actor):

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               epsilon: float,
               step_size: float = 0.1
               ):

    # Get number of states and actions from the environment spec.
    self._num_states = environment_spec.observations.num_values
    self._num_actions = environment_spec.actions.num_values

    # Create the table of Q-values, all initialized at zero.
    self._q = np.zeros((self._num_states, self._num_actions))

    # Store algorithm hyper-parameters.
    self._step_size = step_size
    self._epsilon = epsilon

    # Containers you may find useful.
    self._state = None
    self._action = None
    self._next_state = None

  @property
  def q_values(self):
    return self._q

  def select_action(self, observation):
    return epsilon_greedy(self._q[observation], self._epsilon)

  def observe_first(self, timestep):
    # Set current state.
    self._state = timestep.observation

  def observe(self, action, next_timestep):
    # Unpacking the timestep to lighten notation.
    s = self._state
    a = action
    r = next_timestep.reward
    g = next_timestep.discount
    next_s = next_timestep.observation
    # Compute the action that would be taken from the next state.
    next_a = self.select_action(next_s)
    # Compute the on-policy Q-value update.
    self._action = a
    self._next_state = next_s
    # Compute the on-policy Q-value update.
    self._action = a
    self._next_state = next_s
    # TODO complete the line below to compute the temporal difference error
    # HINT: see step 5 in the pseudocode above.
    self._td_error = r + g * self._q[next_s, next_a] - self._q[s, a]

  def update(self):
    # Optional unpacking to lighten notation.
    s = self._state
    a = self._action
    # Update the Q-value table value at (s, a).
    # TODO: Update the Q-value table value at (s, a).
    # HINT: see step 6 in the pseudocode above, remember that alpha = step_size!
    self._q[s, a] += self._step_size * self._td_error
    # Update the current state.
    self._state = self._next_state