def make_result_matrix(T):
  """
  Create a TxT matrix with values between 0 and 1 to
  be used to compute the metrics.

  Args:
    T : int
      The number of tasks

  Returns:
    result_matrix : numpy.array
      A TxT matrix
  """
  distribution = np.random.rand(T**2)
  # Create a random mask
  mask = np.random.choice([1, 0], distribution.shape, p=[.1, .9]).astype(bool)
  distribution[mask] = np.nan

  result_matrix = []
  count = 0
  for j in range(T):
    temp = []
    for i in range(T):
      temp.append(distribution[count])
      count += 1
    result_matrix.append(temp)
  result_matrix = np.array(result_matrix)

  return result_matrix


# Add event to airtable
atform.add_event('Coding Exercise 4.2: Evaluate your CL algorithm')

set_seed(seed=SEED)
T = len(rehe_accs)  # Number of tasks
## Uncomment below to test you function
result_matrix = make_result_matrix(T)
print(result_matrix)