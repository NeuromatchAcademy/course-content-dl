def ex_net_svd(model, in_dim):
  """
  Performs a Singular Value Decomposition on a given model weights

  Args:
    model: torch.nn.Module
      Neural network model
    in_dim: int
      The input dimension of the model

  Returns:
    U: torch.tensor
      Orthogonal matrix
    Σ: torch.tensor
      Diagonal matrix
    V: torch.tensor
      Orthogonal matrix
  """
  W_tot = torch.eye(in_dim)
  for weight in model.parameters():
    W_tot = weight @ W_tot
  U, Σ, V = torch.svd(W_tot)
  return U, Σ, V


## Uncomment and run
test_net_svd_ex(SEED)