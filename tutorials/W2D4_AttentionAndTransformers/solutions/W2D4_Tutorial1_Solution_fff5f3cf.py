class Transformer(nn.Module):
  """Transformer Encoder network for classification

    Args:
      k (int): Attention embedding size
      heads (int): Number of self attention heads
      depth (int): How many transformer blocks to include
      seq_length (int): How long an input sequence is
      num_tokens (int): Size of dictionary
      num_classes (int): Number of output classes
  """
  def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
    super().__init__()

    self.k = k
    self.num_tokens = num_tokens
    self.token_embedding = nn.Embedding(num_tokens, k)
    self.pos_enc = PositionalEncoding(k)

    transformer_blocks = []
    for i in range(depth):
      transformer_blocks.append(TransformerBlock(k=k, heads=heads))

    self.transformer_blocks = nn.Sequential(*transformer_blocks)
    self.classification_head = nn.Linear(k, num_classes)

  def forward(self, x):
    """Forward pass for Classification Transformer network

    Args:
      x (torch.Tensor): (b, t) sized tensor of tokenized words

    Returns:
      torch.Tensor of size (b, c) with log-probabilities over classes
    """
    x = self.token_embedding(x) * np.sqrt(self.k)
    x = self.pos_enc(x)
    x = self.transformer_blocks(x)

    sequence_avg = x.mean(dim=1)
    x = self.classification_head(sequence_avg)
    logprobs = F.log_softmax(x, dim=1)
    return logprobs