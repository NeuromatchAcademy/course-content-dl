class Seq2SeqEncoder(d2l.Encoder):
  """
  RNN encoder for sequence to sequence learning.
  RNN has the following structure:
  Embedding layer with size vocab_size * embed_size
  RNN layer with size embed_size * num_hiddens * num_layers + dropout
  """

  def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                dropout=0, **kwargs):
    """
    Initialize parameters of Seq2SeqEncoder

    Args:
      num_layers: int
        Number of layers in GRU/RNN
      num_hiddens: int
        Size of hidden layer
      vocab_size: int
        Size of vocabulary
      embed_size: int
        Size of embedding
      dropout: int
        Dropout [default: 0]

    Returns:
      Nothing
    """
    super(Seq2SeqEncoder, self).__init__(**kwargs)

    # Embedding layer
    self.embedding = nn.Embedding(vocab_size, embed_size)
    # Here you're going to implement a GRU as the RNN unit
    self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                      dropout=dropout)

  def forward(self, X, *args):
    """
    Forward pass of Seq2SeqEncoder

    Args:
      X: torch.tensor
        Input features

    Returns:
      output: torch.tensor
        Output with shape (`num_steps`, `batch_size`, `num_hiddens`)
      state: torch.tensor
        State with shape (`num_layers`, `batch_size`, `num_hiddens`)
    """
    # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
    X = self.embedding(X)
    # In RNN models, the first axis corresponds to time steps
    X = X.permute(1, 0, 2)
    # When state is not mentioned, it defaults to zeros, the output should be a RNN function of X!
    output, state = self.rnn(X)
    # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
    # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
    return output, state


# Add event to airtable
atform.add_event('Coding Exercise 3: Encoder')

X = torch.zeros((4, 7), dtype=torch.long)
## uncomment the lines below.
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
output, state = encoder(X)
print(output.shape)
print(state.shape)