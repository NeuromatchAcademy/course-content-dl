class DotProductAttention(nn.Module):
  """ Scaled dot product attention. """

  def __init__(self, dropout, **kwargs):
    """
    Constructs a Scaled Dot Product Attention Instance.

    Args:
      dropout: Integer
        Specifies probability of dropout hyperparameter

    Returns:
      Nothing
    """
    super(DotProductAttention, self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)

  def calculate_score(self, queries, keys):
      """
      Compute the score between queries and keys.

      Args:
      queries: Tensor
        Query is your search tag/Question
        Shape of `queries`: (`batch_size`, no. of queries, head,`k`)
      keys: Tensor
        Descriptions associated with the database for instance
        Shape of `keys`: (`batch_size`, no. of key-value pairs, head, `k`)
      """
      return torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])

  def forward(self, queries, keys, values, b, h, t, k):
    """
    Compute dot products. This is the same operation for each head,
    so we can fold the heads into the batch dimension and use torch.bmm
    Note: .contiguous() doesn't change the actual shape of the data,
    but it rearranges the tensor in memory, which will help speed up the computation
    for this batch matrix multiplication.
    .transpose() is used to change the shape of a tensor. It returns a new tensor
    that shares the data with the original tensor. It can only swap two dimensions.

    Args:
      queries: Tensor
        Query is your search tag/Question
        Shape of `queries`: (`batch_size`, no. of queries, head,`k`)
      keys: Tensor
        Descriptions associated with the database for instance
        Shape of `keys`: (`batch_size`, no. of key-value pairs, head, `k`)
      values: Tensor
        Values are returned results on the query
        Shape of `values`: (`batch_size`, head, no. of key-value pairs,  `k`)
      b: Integer
        Batch size
      h: Integer
        Number of heads
      t: Integer
        Number of keys/queries/values (for simplicity, let's assume they have the same sizes)
      k: Integer
        Embedding size

    Returns:
      out: Tensor
        Matrix Multiplication between the keys, queries and values.
    """
    keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
    queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
    values = values.transpose(1, 2).contiguous().view(b * h, t, k)

    # Matrix Multiplication between the keys and queries
    score = self.calculate_score(queries, keys)  # size: (b * h, t, t)
    softmax_weights = F.softmax(score, dim=2)  # row-wise normalization of weights

    # Matrix Multiplication between the output of the key and queries multiplication and values.
    out = torch.bmm(self.dropout(softmax_weights), values).view(b, h, t, k)  # rearrange h and t dims
    out = out.transpose(1, 2).contiguous().view(b, t, h * k)

    return out