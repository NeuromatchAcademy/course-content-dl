def epsilon_greedy(
    q_values_at_s: np.ndarray,
    epsilon: float = 0.1
    ):
  """
  Return an epsilon-greedy action sample
  Args:
    q_values_at_s: np.ndarray
      Q-values in state s: Q(s, a).
    epsilon: float
      Probability of taking a random action.
  Returns:
    action: Integer
      Chosen action based on epsilon-greedy policy
  """
  # TODO generate a uniform random number and compare it to epsilon to decide if
  # the action should be greedy or not
  # HINT: Use np.random.random() to generate a random float from 0 to 1.
  if epsilon < np.random.random():
    #TODO Greedy: Pick action with the largest Q-value.
    action = np.argmax(q_values_at_s)
  else:
    # Get the number of actions from the size of the given vector of Q-values.
    num_actions = np.array(q_values_at_s).shape[-1]
    # TODO else return a random action
    # HINT: Use np.random.randint() to generate a random integer.
    action = np.random.randint(num_actions)

  return action


# Add event to airtable
atform.add_event('Coding Exercise 5.1: Implement epsilon-greedy')