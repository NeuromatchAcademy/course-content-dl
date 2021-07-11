class TransformerBlock(nn.Module):
  """Transformer Block
  Args:
    k (int): Attention embedding size
    heads (int): number of self-attention heads

  Attributes:
    attention: Multi-head SelfAttention layer
    norm_1, norm_2: LayerNorms
    mlp: feedforward neural network
  """
  def __init__(self, k, heads):
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
    attended = self.attention(x)
    # Complete the input of the first Add & Normalize layer
    x = self.norm_1(attended + x)

    feedforward = self.mlp(x)
    # Complete the input of the second Add & Normalize layer
    x = self.norm_2(feedforward + x)

    return x