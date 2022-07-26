def custom_torch_RSM_fct(features):
  """
  Custom function to calculate representational similarity matrix (RSM) of a feature
  matrix using pairwise cosine similarity.

  Args:
    features: 2D torch.Tensor
      Feature matrix of size (nbr items x nbr features)

  Returns:
    rsm: 2D torch.Tensor
      Similarity matrix of size (nbr items x nbr items)
  """

  num_items, num_features = features.shape

  # EXERCISE: Implement RSM calculation
  rsm = torch.nn.functional.cosine_similarity(
      features.reshape(1, num_items, num_features),
      features.reshape(num_items, 1, num_features),
      dim=2
      )

  if not rsm.shape == (num_items, num_items):
    raise ValueError(f"RSM should be of shape ({num_items}, {num_items})")

  return rsm


# Add event to airtable
atform.add_event('Coding Exercise 2.1.1: Complete a function that calculates RSMs')

## Test implementation by comparing output to solution implementation
test_custom_torch_RSM_fct(custom_torch_RSM_fct)