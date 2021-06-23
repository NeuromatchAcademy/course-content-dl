class LSTM(nn.Module):
  def __init__(self, layers, output_size, hidden_size, vocab_size, embed_size):
    super(LSTM, self).__init__()

    self.output_size = output_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.n_layers = layers

    self.word_embeddings = nn.Embedding(vocab_size, embed_size)
    self.dropout = nn.Dropout(0.5)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=self.n_layers)
    self.fc = nn.Linear(self.n_layers*self.hidden_size, output_size)

  def forward(self, input_sentences):
    # Embeddings
    # `input` shape: (`num_steps`, `batch_size`, `num_hiddens`)
    input = self.word_embeddings(input_sentences).permute(1, 0, 2)

    hidden = (torch.randn(self.n_layers, input.shape[1],
                          self.hidden_size).to(device),
              torch.randn(self.n_layers, input.shape[1],
                          self.hidden_size).to(device))
    # Dropout for regularization
    input = self.dropout(input)
    # LSTM
    output, hidden = self.lstm(input, hidden)

    h_n = hidden[0].permute(1, 0, 2)
    h_n = h_n.contiguous().view(h_n.shape[0], -1)

    logits = self.fc(h_n)

    return logits


sampleLSTM = LSTM(3, 10, 100, 1000, 300)
print(sampleLSTM)