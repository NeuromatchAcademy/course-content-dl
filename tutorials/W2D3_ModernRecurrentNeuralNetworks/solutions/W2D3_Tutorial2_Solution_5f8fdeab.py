class VanillaRNN(nn.Module):
  """
  Vanilla RNN with following structure:
  Embedding of size vocab_size * embed_size # Embedding Layer
  RNN of size embed_size * hidden_size * self.n_layers # RNN Layer
  Linear of size self.n_layers*hidden_size * output_size # Fully connected layer
  """

  def __init__(self, layers, output_size, hidden_size, vocab_size, embed_size,
               device):
    """
    Initialize parameters of VanillaRNN

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
    super(VanillaRNN, self).__init__()
    self.n_layers= layers
    self.hidden_size = hidden_size
    self.device = device
    # Define the embedding
    self.embeddings = nn.Embedding(vocab_size, embed_size)
    # Define the RNN layer
    self.rnn = nn.RNN(embed_size, hidden_size, self.n_layers)
    # Define the fully connected layer
    self.fc = nn.Linear(self.n_layers *hidden_size, output_size)

  def forward(self, inputs):
    """
    Forward pass of VanillaRNN

    Args:
      inputs: torch.tensor
        Input features

    Returns:
      logits: torch.tensor
        Output of final fully connected layer
    """
    input = self.embeddings(inputs)
    input = input.permute(1, 0, 2)
    h_0 = torch.zeros(2, input.size()[1], self.hidden_size).to(self.device)
    output, h_n = self.rnn(input, h_0)
    h_n = h_n.permute(1, 0, 2)
    # Reshape the data and create a copy of the tensor such that the
    # order of its elements in memory is the same as if it had been created
    # from scratch with the same data. Without contiguous it may raise an error
    # RuntimeError: input is not contiguous;
    # Note that this is necessary as permute may return a non-contiguous tensor
    h_n = h_n.contiguous().reshape(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
    logits = self.fc(h_n)

    return logits


# Add event to airtable
atform.add_event('Coding Exercise 1.1: Vanilla RNN')

## Uncomment to test VanillaRNN class
sampleRNN = VanillaRNN(2, 10, 50, 1000, 300, DEVICE)
print(sampleRNN)