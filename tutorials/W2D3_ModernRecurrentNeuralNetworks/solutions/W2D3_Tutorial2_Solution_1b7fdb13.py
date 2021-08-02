class biLSTM(nn.Module):
  def __init__(self, output_size, hidden_size, vocab_size, embed_size,
               device):
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
    # Define the fully-connected layer
    self.fc = nn.Linear(4*hidden_size, output_size)


  def forward(self, input_sentences):
    input = self.word_embeddings(input_sentences).permute(1, 0, 2)
    hidden = (torch.randn(4, input.shape[1], self.hidden_size).to(self.device),
              torch.randn(4, input.shape[1], self.hidden_size).to(self.device))
    input = self.dropout(input)

    output, hidden = self.bilstm(input, hidden)

    h_n = hidden[0].permute(1, 0, 2)
    h_n = h_n.contiguous().view(h_n.shape[0], -1)
    logits = self.fc(h_n)

    return logits


## Uncomment to run
sampleBiLSTM = biLSTM(10, 100, 1000, 300, DEVICE)
print(sampleBiLSTM)