class NeuralNet(nn.Module):
  """ A vanilla neural network. """
  def __init__(self, batch_size, output_size, hidden_size, vocab_size,
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

    self.batch_size = batch_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_length = embedding_length

    self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
    self.word_embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
    self.fc1 = nn.Linear(embedding_length, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)

  def forward(self, inputs):
    """
    Compute the final labels by taking tokens as input.

    Args:
      inputs: Tensor
        Tensor of tokens in the text

    Returns:
      out: Tensor
        Final prediction Tensor
    """
    input = self.word_embeddings(inputs)  # convert text to embeddings
    # Average the word embedddings in a sentence
    # Use torch.nn.functional.avg_pool2d to compute the averages
    pooled = F.avg_pool2d(input, (input.shape[1], 1)).squeeze(1)
    # Pass the embeddings through the neural net
    # Use ReLU as the non-linearity
    x = self.fc1(pooled)
    x = F.relu(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output


# add event to airtable
atform.add_event('Coding Exercise 1: Neural Net for text classification')