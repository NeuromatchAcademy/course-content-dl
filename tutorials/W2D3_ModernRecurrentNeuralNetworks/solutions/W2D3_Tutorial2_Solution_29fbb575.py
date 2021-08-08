class Seq2SeqEncoder(d2l.Encoder):
  """The RNN encoder for sequence to sequence learning."""
  def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                dropout=0, **kwargs):
    super(Seq2SeqEncoder, self).__init__(**kwargs)

    # Embedding layer
    self.embedding = nn.Embedding(vocab_size, embed_size)
    # Here you're going to implement a GRU as the RNN unit
    self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                      dropout=dropout)

  def forward(self, X, *args):
    # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
    X = self.embedding(X)
    # In RNN models, the first axis corresponds to time steps
    X = X.permute(1, 0, 2)
    # When state is not mentioned, it defaults to zeros, the output should be a RNN function of X!
    output, state = self.rnn(X)
    # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
    # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
    return output, state


# add event to airtable
atform.add_event('Coding Exercise 3: Encoder')

X = torch.zeros((4, 7), dtype=torch.long)
## uncomment the lines below.
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
output, state = encoder(X)
print(output.shape)
print(state.shape)