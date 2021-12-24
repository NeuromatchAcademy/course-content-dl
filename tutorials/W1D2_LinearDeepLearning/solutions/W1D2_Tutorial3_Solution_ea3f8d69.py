def ex_net_rsm(h):
  """Calculates the Representational Similarity Matrix

  Arg:
    h (torch.Tensor): activity of a hidden layer

  Returns:
    (torch.Tensor): Representational Similarity Matrix
  """
  rsm = h @ h.T
  return rsm

# add event to airtable
atform.add_event(' Coding Exercise 3: RSA')

## Uncomment and run
test_net_rsm_ex(SEED)