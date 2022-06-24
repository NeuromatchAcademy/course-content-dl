class Encoder(nn.Module):
  """The RNN encoder for sequence to sequence learning."""
  def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                dropout=0):
    super(Encoder, self).__init__()
    # Embedding layer
    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.rnn = nn.RNN(embed_size, num_hiddens, num_layers,
                      dropout=dropout)

  def forward(self, X):
    # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
    X = self.embedding(X)
    # In RNN models, the first axis corresponds to time steps
    X = X.permute(1, 0, 2)
    # When state is not mentioned, it defaults to zeros
    output, state = self.rnn(X)
    # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
    # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
    return output, state


class Decoder(nn.Module):
  """The RNN decoder for sequence to sequence learning."""
  def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                dropout=0):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embed_size)

    self.rnn = nn.RNN(embed_size + num_hiddens, num_hiddens, num_layers,
                      dropout=dropout)
    self.dense = nn.Linear(num_hiddens, vocab_size)
    self.dropout = nn.Dropout(0.25)

  def init_state(self, enc_outputs):
    return enc_outputs[1]

  def forward(self, X, state):
    """Hint: always make sure your sizes are correct"""
    # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
    X = self.embedding(X).permute(1, 0, 2)
    # Broadcast `context` so it has the same `num_steps` as `X`
    context = state[-1].repeat(X.shape[0], 1, 1)

    # Concatenate X and context
    X_and_context = torch.cat((X, context), 2)

    # Recurrent unit
    output, state = self.rnn(X_and_context, state)
    # Linear layer
    output = self.dense(output).permute(1, 0, 2)
    # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)

    return output, state


## Uncomment to run
encoder = Encoder(1000, 300, 100, 2, 0.1)
decoder = Decoder(1000, 300, 100, 2, 0.1)
print(encoder)
print(decoder)

# add event to airtable
atform.add_event('Coding Exercise 5: Implement Encoder and Decoder')