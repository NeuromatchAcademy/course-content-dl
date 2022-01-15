class SelfAttention(nn.Module):
  """  Multi-head self attention layer. """

  def __init__(self, k, heads=8, dropout=0.1):
    """
    Initiates the following attributes:
    to_keys: Transforms input to k x k*heads key vectors
    to_queries: Transforms input to k x k*heads query vectors
    to_values: Transforms input to k x k*heads value vectors
    unify_heads: combines queries, keys and values to a single vector

    Args:
      k: Integer
        Size of attention embeddings
      heads: Integer
        Number of attention heads

    Returns:
      Nothing
    """
    super().__init__()
    self.k, self.heads = k, heads

    self.to_keys = nn.Linear(k, k * heads, bias=False)
    self.to_queries = nn.Linear(k, k * heads, bias=False)
    self.to_values = nn.Linear(k, k * heads, bias=False)
    self.unify_heads = nn.Linear(k * heads, k)

    self.attention = DotProductAttention(dropout)

  def forward(self, x):
    """
    Implements forward pass of self-attention layer

    Args:
      x: Tensor
        Batch x t x k sized input

    Returns:
      unify_heads: Tensor
        Self-attention based unified Query/Value/Key tensors
    """
    b, t, k = x.size()
    h = self.heads

    # We reshape the queries, keys and values so that each head has its own dimension
    queries = self.to_queries(x).view(b, t, h, k)
    keys = self.to_keys(x).view(b, t, h, k)
    values = self.to_values(x).view(b, t, h, k)

    out = self.attention(queries, keys, values, b, h, t, k)

    return self.unify_heads(out)


# add event to airtable
atform.add_event('Coding Exercise 5: Q, K, V attention')