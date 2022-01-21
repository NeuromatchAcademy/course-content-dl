class NeuralNet(nn.Module):
  """
  Neural Network with following structure:
  nn.Embedding(vocab_size, embedding_length)
  + nn.Parameter(word_embeddings, requires_grad=False) # Embedding Layer
  nn.Linear(embedding_length, hidden_size) # Fully connected layer #1
  nn.Linear(hidden_size, output_size) # Fully connected layer #2
  """

  def __init__(self, output_size, hidden_size, vocab_size, embedding_length,
               word_embeddings):
    """
    Initialize parameters of NeuralNet

    Args:
      output_size: int
        Size of final fully connected layer
      hidden_size: int
        Size of hidden/first fully connected layer
      vocab_size: int
        Size of vocabulary
      embedding_length: int
        Length of embedding
      word_embeddings: TEXT.vocab.vectors instance
        Word Embeddings

    Returns:
      Nothing
    """
    super(NeuralNet, self).__init__()

    self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
    self.word_embeddings.weight = nn.Parameter(word_embeddings,
                                               requires_grad=False)
    self.fc1 = nn.Linear(embedding_length, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)


  def forward(self, inputs):
    """
    Forward pass of NeuralNet

    Args:
      Inputs: list
        Text

    Returns:
      output: torch.tensor
        Outputs/Predictions
    """
    input = self.word_embeddings(inputs)  # Convert text to embeddings
    # Average the word embeddings in a sentence
    # Use torch.nn.functional.avg_pool2d to compute the averages
    pooled = F.avg_pool2d(input, (input.shape[1], 1)).squeeze(1)

    # Pass the embeddings through the neural net
    # A fully-connected layer
    x = self.fc1(pooled)

    # ReLU activation
    x = F.relu(x)
    # Another fully-connected layer
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)

    return output


# Add event to airtable
atform.add_event('Coding Exercise 3.1: Simple Feed Forward Net')

# Uncomment to check your code
nn_model = NeuralNet(2, 128, 100, 300, TEXT.vocab.vectors)
print(nn_model)