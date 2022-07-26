def custom_simclr_contrastive_loss(proj_feat1, proj_feat2, temperature=0.5):
  """
  Returns contrastive loss, given sets of projected features, with positive
  pairs matched along the batch dimension.

  Args:
    Required:
      proj_feat1: 2D torch.Tensor
        Projected features for first image with augmentations (size: batch_size x feat_size)
      proj_feat2: 2D torch.Tensor
        Projected features for second image with augmentations (size: batch_size x feat_size)
    Optional:
      temperature: Float
        Relaxation temperature (default: 0.5)
        l2 normalization along with temperature effectively weights different
        examples, and an appropriate temperature can help the model learn from hard negatives.
  Returns:
    loss: Float
      Mean contrastive loss
  """
  device = proj_feat1.device

  if len(proj_feat1) != len(proj_feat2):
    raise ValueError(f"Batch dimension of proj_feat1 ({len(proj_feat1)}) "
                     f"and proj_feat2 ({len(proj_feat2)}) should be same")

  batch_size = len(proj_feat1) # N
  z1 = torch.nn.functional.normalize(proj_feat1, dim=1)
  z2 = torch.nn.functional.normalize(proj_feat2, dim=1)

  proj_features = torch.cat([z1, z2], dim=0) # 2N x projected feature dimension
  similarity_matrix = torch.nn.functional.cosine_similarity(
      proj_features.unsqueeze(1), proj_features.unsqueeze(0), dim=2
      ) # dim: 2N x 2N

  # Initialize arrays to identify sets of positive and negative examples, of
  # shape (batch_size * 2, batch_size * 2), and where
  # 0 indicates that 2 images are NOT a pair (either positive or negative, depending on the indicator type)
  # 1 indices that 2 images ARE a pair (either positive or negative, depending on the indicator type)
  pos_sample_indicators = torch.roll(torch.eye(2 * batch_size), batch_size, 1).to(device)
  neg_sample_indicators = (torch.ones(2 * batch_size) - torch.eye(2 * batch_size)).to(device)

  # Implement the SimClr loss calculation
  # Calculate the numerator of the Loss expression by selecting the appropriate elements from similarity_matrix.
  # Use the pos_sample_indicators tensor
  numerator = torch.exp(similarity_matrix / temperature)[pos_sample_indicators.bool()]

  # Calculate the denominator of the Loss expression by selecting the appropriate elements from similarity_matrix,
  # and summing over pairs for each item.
  # Use the neg_sample_indicators tensor
  denominator = torch.sum(
      torch.exp(similarity_matrix / temperature) * neg_sample_indicators,
      dim=1
      )

  if (denominator < 1e-8).any(): # Clamp to avoid division by 0
    denominator = torch.clamp(denominator, 1e-8)

  loss = torch.mean(-torch.log(numerator / denominator))

  return loss


# Add event to airtable
atform.add_event('Coding Exercise 6.1.1: Complete a SimCLR loss function')

## Uncomment below to test your function
test_custom_contrastive_loss_fct(custom_simclr_contrastive_loss)