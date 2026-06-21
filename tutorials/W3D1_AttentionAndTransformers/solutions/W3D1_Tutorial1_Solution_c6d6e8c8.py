class Transformer(nn.Module):
  """ Transformer Encoder network for classification. """

  def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
    """
    Initiates the Transformer Network

    Args:
      k: Integer
        Attention embedding size
      heads: Integer
        Number of self attention heads
      depth: Integer
        Number of Transformer Blocks
      seq_length: Integer
        Length of input sequence
      num_tokens: Integer
        Size of dictionary
      num_classes: Integer
        Number of output classes

    Returns:
      Nothing
    """
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
    """
    Forward pass for Classification within Transformer network

    Args:
      x: Tensor
        (b, t) sized tensor of tokenized words

    Returns:
      logprobs: Tensor
        Log-probabilities over classes sized (b, c)
    """
    x = self.token_embedding(x) * np.sqrt(self.k)
    x = self.pos_enc(x)
    x = self.transformer_blocks(x)

    sequence_avg = x.mean(dim=1)
    x = self.classification_head(sequence_avg)
    logprobs = F.log_softmax(x, dim=1)
    return logprobs