class DQN(acme.Actor):
  """
  Implementation of a Deep Q Network Agent
  """

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               network: nn.Module,
               replay_capacity: int = 100_000,
               epsilon: float = 0.1,
               batch_size: int = 1,
               learning_rate: float = 5e-4,
               target_update_frequency: int = 10):
    """
    DQN Based Agent Initialisation

    Args:
      environment_spec: specs.EnvironmentSpec
        * actions: DiscreteArray(shape=(), dtype=int32, name=action, minimum=0, maximum=3, num_values=4)
        * observations: Array(shape=(9, 10, 3), dtype=dtype('float32'), name='observation_grid')
        * rewards: Array(shape=(), dtype=dtype('float32'), name='reward')
        * discounts: BoundedArray(shape=(), dtype=dtype('float32'), name='discount', minimum=0.0, maximum=1.0)
      network: nn.Module
        Deep Q Network
      replay_capacity: int
        Capacity of the replay buffer [default: 100000]
      epsilon: float
        Controls the exploration-exploitation tradeoff
      batch_size: int
        Batch Size [default = 1]
      learning_rate: float
        Rate at which the neural fitted agent learns [default = 3e-4]
      target_update_frequency: int
        Frequency with which target network is updated

    Returns:
      Nothing
    """
    # Store agent hyperparameters and network.
    self._num_actions = environment_spec.actions.num_values
    self._epsilon = epsilon
    self._batch_size = batch_size
    self._q_network = q_network

    # Create a second q net with the same structure and initial values, which
    # we'll be updating separately from the learned q-network.
    self._target_network = copy.deepcopy(self._q_network)

    # Container for the computed loss (see run_loop implementation above).
    self.last_loss = 0.0

    # Create the replay buffer.
    self._replay_buffer = ReplayBuffer(replay_capacity)
    # Keep an internal tracker of steps
    self._current_step = 0

    # How often to update the target network
    self._target_update_frequency = target_update_frequency
    # Setup optimizer that will train the network to minimize the loss.
    self._optimizer = torch.optim.Adam(self._q_network.parameters(), lr=learning_rate)
    self._loss_fn = nn.MSELoss()

  def select_action(self, observation):
    """
    Action Selection Algorithm

    Args:
      observation: enum
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
      action: Integer
        Chosen random action
    """
    # Compute Q-values.
    # Sonnet requires a batch dimension, which we squeeze out right after.
    q_values = self._q_network(torch.tensor(observation).unsqueeze(0))  # Adds batch dimension.
    q_values = q_values.squeeze(0)   # Removes batch dimension

    # Select epsilon-greedy action.
    if self._epsilon < torch.rand(1):
      action = q_values.argmax(axis=-1)
    else:
      action = torch.randint(low=0, high=self._num_actions , size=(1,), dtype=torch.int64)
    return action

  def q_values(self, observation):
    q_values = self._q_network(torch.tensor(observation).unsqueeze(0))
    return q_values.squeeze(0).detach()

  def update(self):
    """
    Updates replay buffer

    Args:
      None

    Returns:
      Nothing
    """
    self._current_step += 1

    if not self._replay_buffer.is_ready(self._batch_size):
      # If the replay buffer is not ready to sample from, do nothing.
      return

    # Sample a minibatch of transitions from experience replay.
    transitions = self._replay_buffer.sample(self._batch_size)

    # Optionally unpack the transitions to lighten notation.
    # Note: each of these tensors will be of shape [batch_size, ...].
    s = torch.tensor(transitions.state)
    a = torch.tensor(transitions.action,dtype=torch.int64)
    r = torch.tensor(transitions.reward)
    d = torch.tensor(transitions.discount)
    next_s = torch.tensor(transitions.next_state)

    # Compute the Q-values at next states in the transitions.
    with torch.no_grad():
      #TODO get the value of the next states evaluated by the target network
      #HINT: use self._target_network, defined above.
      q_next_s = self._target_network(next_s)  # Shape [batch_size, num_actions].
      max_q_next_s = q_next_s.max(axis=-1)[0]
      # Compute the TD error and then the losses.
      target_q_value = r + d * max_q_next_s

    # Compute the Q-values at original state.
    q_s = self._q_network(s)

    # Gather the Q-value corresponding to each action in the batch.
    q_s_a = q_s.gather(1, a.view(-1,1)).squeeze(0)

    # Average the squared TD errors over the entire batch
    loss = self._loss_fn(target_q_value, q_s_a)

    # Compute the gradients of the loss with respect to the q_network variables.
    self._optimizer.zero_grad()

    loss.backward()
    # Apply the gradient update.
    self._optimizer.step()

    if self._current_step % self._target_update_frequency == 0:
      self._target_network.load_state_dict(self._q_network.state_dict())
    # Store the loss for logging purposes (see run_loop implementation above).
    self.last_loss = loss.detach().numpy()

  def observe_first(self, timestep: dm_env.TimeStep):
    self._replay_buffer.add_first(timestep)

  def observe(self, action: int, next_timestep: dm_env.TimeStep):
    self._replay_buffer.add(action, next_timestep)


# Add event to airtable
atform.add_event('Coding Exercise 7.1: Run a DQN Agent')

# Create a convenient container for the SARS tuples required by NFQ.
Transitions = collections.namedtuple(
    'Transitions', ['state', 'action', 'reward', 'discount', 'next_state'])