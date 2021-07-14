def ex_net_rsm(h):
  """Calculates the Representational Similarity Matrix

  Arg:
    h (torch.Tensor): activity of a hidden layer

  Returns:
    (torch.Tensor): Representational Similarity Matrix
  """
  rsm = h @ h.T
  return rsm


test_net_rsm_ex()