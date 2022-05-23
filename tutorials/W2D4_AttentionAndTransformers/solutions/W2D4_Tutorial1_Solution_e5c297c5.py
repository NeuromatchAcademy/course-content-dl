class TransformerBlock(nn.Module):
  """ Block to instantiate transformers. """

  def __init__(self, k, heads):
    """
    Initiates following attributes
    attention: Initiating Multi-head Self-Attention layer
    norm1, norm2: Initiating Layer Norms
    mlp: Initiating Feed Forward Neural Network

    Args:
      k: Integer
        Attention embedding size
      heads: Integer
        Number of self-attention heads

    Returns:
      Nothing
    """
    super().__init__()

    self.attention = SelfAttention(k, heads=heads)

    self.norm_1 = nn.LayerNorm(k)
    self.norm_2 = nn.LayerNorm(k)

    hidden_size = 2 * k  # This is a somewhat arbitrary choice
    self.mlp = nn.Sequential(
        nn.Linear(k, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, k))

  def forward(self, x):
    """
    Defines the network structure and flow across a subset of transformer blocks

    Args:
      x: Tensor
        Input Sequence to be processed by the network

    Returns:
      x: Tensor
        Input post-processing by add and normalise blocks [See Architectural Block above for visual details]
    """
    attended = self.attention(x)
    # Complete the input of the first Add & Normalize layer
    x = self.norm_1(attended + x)

    feedforward = self.mlp(x)
    # Complete the input of the second Add & Normalize layer
    x = self.norm_2(feedforward + x)

    return x


# add event to airtable
atform.add_event('Coding Exercise 4: Transformer encoder')