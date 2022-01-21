class LSTM(nn.Module):
  """
  LSTM (Long Short Term Memory) with following structure
  Embedding layer of size vocab_size * embed_size
  Dropout layer with dropout_probability of 0.5
  LSTM layer of size embed_size * hidden_size * num_layers
  Fully connected layer of n_layers*hidden_size * output_size
  """

  def __init__(self, layers, output_size, hidden_size, vocab_size, embed_size,
               device):
    """
    Initialize parameters of LSTM

    Args:
      layers: int
        Number of layers
      output_size: int
        Size of final fully connected layer
      hidden_size: int
        Size of hidden layer
      vocab_size: int
        Size of vocabulary
      device: string
        GPU if available, CPU otherwise
      embed_size: int
        Size of embedding

    Returns:
      Nothing
    """
    super(LSTM, self).__init__()
    self.n_layers = layers
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.device = device
    # Define the word embeddings
    self.word_embeddings = nn.Embedding(vocab_size, embed_size)
    # Define the dropout layer
    self.dropout = nn.Dropout(0.5)
    # Define the lstm layer
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=self.n_layers)
    # Define the fully-connected layer
    self.fc = nn.Linear(self.n_layers*self.hidden_size, output_size)


  def forward(self, input_sentences):
    """
    Forward pass of LSTM
    Hint: Make sure the shapes of your tensors match the requirement

    Args:
      input_sentences: torch.tensor
        Input Sentences

    Returns:
      logits: torch.tensor
        Output of final fully connected layer
    """
    # Embeddings
    # `input` shape: (`num_steps`, `batch_size`, `num_hiddens`)
    input = self.word_embeddings(input_sentences).permute(1, 0, 2)

    hidden = (torch.randn(self.n_layers, input.shape[1],
                          self.hidden_size).to(self.device),
              torch.randn(self.n_layers, input.shape[1],
                          self.hidden_size).to(self.device))
    # Dropout for regularization
    input = self.dropout(input)
    # LSTM
    output, hidden = self.lstm(input, hidden)

    h_n = hidden[0].permute(1, 0, 2)
    h_n = h_n.contiguous().view(h_n.shape[0], -1)

    logits = self.fc(h_n)

    return logits


# Add event to airtable
atform.add_event('Coding Exercise 2.1: Implementing LSTM')

## Uncomment to run
sampleLSTM = LSTM(3, 10, 100, 1000, 300, DEVICE)
print(sampleLSTM)