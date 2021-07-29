def ex_net_svd(model, in_dim):
  """Performs a Singular Value Decomposition on a given model weights

  Args:
    model (torch.nn.Module): neural network model
    in_dim (int): the input dimension of the model

  Returns:
    U, Σ, V (Tensors): Orthogonal, diagonal, and orthogonal matrices
  """
  W_tot = torch.eye(in_dim)
  for weight in model.parameters():
    W_tot = weight @ W_tot
  U, Σ, V = torch.svd(W_tot)
  return U, Σ, V

#add event to airtable
atform.add_event('Coding Exercise 2: SVD')


## Uncomment and run
test_net_svd_ex(SEED)