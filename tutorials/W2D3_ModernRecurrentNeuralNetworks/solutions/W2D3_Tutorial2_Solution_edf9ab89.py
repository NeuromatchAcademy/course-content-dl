class biLSTM(nn.Module):
  """
  Bidirectional LSTM with following structure
  Embedding layer of size vocab_size * embed_size
  Dropout layer with dropout_probability of 0.5
  biLSTM layer of size embed_size * hidden_size * num_layers
  Fully connected layer of n_layers*hidden_size * output_size
  """

  def __init__(self, output_size, hidden_size, vocab_size, embed_size,
               device):
    """
    Initialize parameters of biLSTM

    Args:
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
    super(biLSTM, self).__init__()
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.device = device
    # Define the word embeddings
    self.word_embeddings = nn.Embedding(vocab_size, embed_size)
    # Define the dropout layer
    self.dropout = nn.Dropout(0.5)
    # Define the bilstm layer
    self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers=2, bidirectional=True)
    # Define the fully-connected layer; 4 = 2*2: 2 for stacking and 2 for bidirectionality
    self.fc = nn.Linear(4*hidden_size, output_size)


  def forward(self, input_sentences):
    """
    Forward pass of biLSTM

    Args:
      input_sentences: torch.tensor
        Input Sentences

    Returns:
      logits: torch.tensor
        Output of final fully connected layer
    """
    input = self.word_embeddings(input_sentences).permute(1, 0, 2)
    hidden = (torch.randn(4, input.shape[1], self.hidden_size).to(self.device),
              torch.randn(4, input.shape[1], self.hidden_size).to(self.device))
    input = self.dropout(input)

    output, hidden = self.bilstm(input, hidden)

    h_n = hidden[0].permute(1, 0, 2)
    h_n = h_n.contiguous().view(h_n.shape[0], -1)
    logits = self.fc(h_n)

    return logits


# Add event to airtable
atform.add_event('Coding Exercise 2.2: BiLSTM')

## Uncomment to run
sampleBiLSTM = biLSTM(10, 100, 1000, 300, DEVICE)
print(sampleBiLSTM)