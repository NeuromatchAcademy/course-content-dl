def custom_torch_RSM_fct(features):
  """
  custom_torch_RSM_fct(features)

  Custom function to calculates representational similarity matrix (RSM) of a feature
  matrix using pairwise cosine similarity.

  Required args:
  - features (2D torch Tensor): feature matrix (items x features)

  Returns:
  - rsm (2D torch Tensor): similarity matrix
      (nbr features items x nbr features items)
  """
  # EXERCISE: Implement RSM calculation
  rsm = torch.nn.functional.cosine_similarity(features.unsqueeze(1),
                                              features.unsqueeze(0), dim=2)

  if not rsm.shape == (len(features), len(features)):
    raise ValueError(
        f"RSM should be of shape ({len(features)}, {len(features)})"
        )

  return rsm

# Test implementation by comparing output to solution implementation
test_custom_torch_RSM_fct(custom_torch_RSM_fct)