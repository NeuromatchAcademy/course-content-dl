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
  result_matrix = []
  # Uniform sampled distribution
  distribution = np.random.choice([1, 0], T, p=[.1, .9])
  place_holder = np.random.randn(T)
  place_holder[distribution] = np.nan # Masking

  # This block is to un-flatten the 25 element matrix into a 5*5 matrix
  for j in range(T):
    temp = []
    for i in range(T):
      temp.append(place_holder[i])
    result_matrix.append(temp)

  result_matrix = np.array(result_matrix)

  return result_matrix


# add event to airtable
atform.add_event('Coding Exercise 4.2: Evaluate your CL algorithm')

T = len(rehe_accs)  # number of tasks
## Uncommnet below to test you function
result_matrix = make_result_matrix(T)
print(result_matrix)