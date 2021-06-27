def custom_simclr_contrastive_loss(proj_feat1, proj_feat2, temperature=0.5,
                                   seed=SEED):
  """
  custom_simclr_contrastive_loss(proj_feat1, proj_feat2)
  Returns contrastive loss, given sets of projected features, with positive
  pairs matched along the batch dimension.
  Required args:
  - proj_feat1 (2D torch Tensor): first set of projected features
      (batch_size x feat_size)
  - proj_feat2 (2D torch Tensor): second set of projected features
      (batch_size x feat_size)

  Optional args:
  - temperature (float): relaxation temperature. (default: 0.5)
  Returns:
  - loss (float): mean contrastive loss
  """
  # call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)
  device = proj_feat1.device

  if len(proj_feat1) != len(proj_feat2):
    raise ValueError(f"Batch dimension of proj_feat1 ({len(proj_feat1)}) "
                     f"and proj_feat2 ({len(proj_feat2)}) should be same")

  batch_size = len(proj_feat1) # N
  z1 = torch.nn.functional.normalize(proj_feat1, dim=1)
  z2 = torch.nn.functional.normalize(proj_feat2, dim=1)

  proj_features = torch.cat([z1, z2], dim=0) # 2N x projected feature dimension
  similarity_mat = torch.nn.functional.cosine_similarity(
      proj_features.unsqueeze(1), proj_features.unsqueeze(0), dim=2
      ) # dim: 2N x 2N

  # initialize arrays to identify sets of positive and negative examples
  pos_sample_indicators = torch.roll(torch.eye(2 * batch_size), batch_size, 1)
  neg_sample_indicators = torch.ones(2 * batch_size) - torch.eye(2 * batch_size)

  # EXERCISE: Implement the SimClr loss calculation
  # Calculate the numerator of the Loss expression by selecting the appropriate elements from similarity_mat
  # Use the pos_sample_indicators tensor
  numerator = torch.sum(
      torch.exp(similarity_mat / temperature) * pos_sample_indicators.to(device),
      dim=1
      )
  # Calculate the denominator of the Loss expression by selecting the appropriate elements from similarity_mat
  # Use the neg_sample_indicators tensor
  denominator = torch.sum(
      torch.exp(similarity_mat / temperature) * neg_sample_indicators.to(device),
      dim=1
      )

  if (denominator < 1e-8).any(): # clamp to avoid division by 0
    denominator = torch.clamp(denominator, 1e-8)

  loss = torch.mean(-torch.log(numerator / denominator))

  return loss


## Uncomment below to test your function
test_custom_contrastive_loss_fct(custom_simclr_contrastive_loss)