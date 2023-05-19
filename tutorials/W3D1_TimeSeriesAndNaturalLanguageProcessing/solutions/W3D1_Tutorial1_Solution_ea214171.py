class NeuralNet(nn.Module):
  """ A vanilla neural network. """
  def __init__(self, output_size, hidden_size, vocab_size,
               embedding_length, word_embeddings):
    """
    Constructs a vanilla Neural Network Instance.

    Args:
      batch_size: Integer
        Specifies probability of dropout hyperparameter
      output_size: Integer
        Specifies the size of output vector
      hidden_size: Integer
        Specifies the size of hidden layer
      vocab_size: Integer
        Specifies the size of the vocabulary
        i.e. the number of tokens in the vocabulary
      embedding_length: Integer
        Specifies the size of the embedding vector
      word_embeddings
        Specifies the weights to create embeddings from
        voabulary.

    Returns:
      Nothing
    """
    super(NeuralNet, self).__init__()

    self.output_size = output_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_length = embedding_length

    # self.word_embeddings = nn.EmbeddingBag(vocab_size, embedding_length, sparse=False)
    self.word_embeddings = nn.EmbeddingBag.from_pretrained(embedding_fasttext.vectors)
    self.word_embeddings.weight.requiresGrad = False
    self.fc1 = nn.Linear(embedding_length, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.init_weights()

  def init_weights(self):
      initrange = 0.5
      # self.word_embeddings.weight.data.uniform_(-initrange, initrange)
      self.fc1.weight.data.uniform_(-initrange, initrange)
      self.fc1.bias.data.zero_()
      self.fc2.weight.data.uniform_(-initrange, initrange)
      self.fc2.bias.data.zero_()

  def forward(self, inputs, offsets):
    """
    Compute the final labels by taking tokens as input.

    Args:
      inputs: Tensor
        Tensor of tokens in the text

    Returns:
      out: Tensor
        Final prediction Tensor
    """
    embedded = self.word_embeddings(inputs, offsets)  # convert text to embeddings
    # Pass the embeddings through the neural net
    # Use ReLU as the non-linearity
    x = self.fc1(embedded)
    x = F.relu(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output