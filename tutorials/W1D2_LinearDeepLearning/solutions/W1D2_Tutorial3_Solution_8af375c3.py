def ex_initializer_(model, gamma=1e-12):
  """(in-place) Re-initialization of weights

  Args:
    model (torch.nn.Module): PyTorch neural net model
    gamma (float): initialization scale
  """
  for weight in model.parameters():
    n_out, n_in = weight.shape
    sigma = gamma / math.sqrt(n_in + n_out)
    nn.init.normal_(weight, mean=0.0, std=sigma)


test_initializer_ex()