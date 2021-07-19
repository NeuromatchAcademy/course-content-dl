def functionA(A, B):
  """
  This function takes in two 2D tensors A and B and returns the column sum of
  A multiplied by the sum of all the elmements of B, i.e., a scalar.

  Args:
    A: torch.Tensor
    B: torch.Tensor
  Retuns:
    output: torch.Tensor
      The multiplication of the column sum of `A` by the sum of `B`.
  """
  # TODO multiplication the sum of the tensors
  output = A.sum(axis=0) * B.sum()
  return output


def functionB(C):
  """
  This function takes in a square matrix C and returns a 2D tensor consisting of
  a flattened C with the index of each element appended to this tensor in the
  row dimension.

  Args:
    C: torch.Tensor
  Retuns:
    output: torch.Tensor
      Concatenated tensor.
  """
  # TODO flatten the tensor  C
  C = C.flatten()
  # TODO create the idx tensor to be concatenated to C
  idx_tensor = torch.arange(0, len(C))
  # TODO concatenate the two tensors
  output = torch.cat([idx_tensor.unsqueeze(1), C.unsqueeze(1)], axis=1)

  output = torch.zeros(1)
  return output


def functionC(D, E):
  """
  This function takes in two 2D tensors D and E . If the dimensions allow it,
  this function returns the elementwise sum of D-shaped E, and D;
  else this function returns a 1D tensor that is the concatenation of the
  two tensors.

  Args:
    D: torch.Tensor
    E: torch.Tensor
  Retuns:
    output: torch.Tensor
      Concatenated tensor.
  """
  # TODO check we can reshape E into the shape of D
  if torch.numel(D) == torch.numel(E):
    # TODO reshape E into the shape of D
    E = E.reshape(D.shape)
    # TODO sum the two tensors
    output = D + E
  else:
    # TODO flatten both tensors
    D = D.reshape(1, -1)
    E = E.reshape(1, -1)
    # TODO concatenate the two tensors in the correct dimension
    output = torch.cat([D, E], axis=1)

  return output


## Implement the functions above and then uncomment the following lines to test your code
print(functionA(torch.tensor([[1, 1], [1, 1]]), torch.tensor([[1, 2, 3], [1, 2, 3]])))
print(functionB(torch.tensor([[2, 3], [-1, 10]])))
print(functionC(torch.tensor([[1, -1], [-1, 3]]), torch.tensor([[2, 3, 0, 2]])))
print(functionC(torch.tensor([[1, -1], [-1, 3]]), torch.tensor([[2, 3, 0]])))