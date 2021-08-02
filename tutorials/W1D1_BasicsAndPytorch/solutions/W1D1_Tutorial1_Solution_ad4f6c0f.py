def tensor_creation(Z):
  """A function that creates various tensors.

  Args:
    Z (numpy.ndarray): An array of shape

  Returns:
    A : 20 by 21 tensor consisting of ones
    B : a tensor with elements equal to the elements of numpy array  Z
    C : a tensor with the same number of elements as A but with values ∼U(0,1)
    D : a 1D tensor containing the even numbers between 4 and 40 inclusive.
  """

  A = torch.ones(20, 21)
  B = torch.tensor(Z)
  C = torch.rand_like(A)
  D = torch.arange(4, 41, step=2)

  return A, B, C, D

# add timing to airtable
atform.add_event('Coding Exercise 2.1: Creating Tensors')

# numpy array to copy later
Z = np.vander([1, 2, 3], 4)

# Uncomment below to check your function!
A, B, C, D = tensor_creation(Z)
checkExercise1(A, B, C, D)