class VanillaRNN(nn.Module):
  def __init__(self, output_size, hidden_size, vocab_size, embed_size):
    super(VanillaRNN, self).__init__()

    self.hidden_size = hidden_size
    self.embeddings = nn.Embedding(vocab_size,embed_size)
    self.rnn = nn.RNN(embed_size, hidden_size, num_layers=2)
    self.fc = nn.Linear(2*hidden_size, output_size)

  def forward(self, inputs):
    input = self.embeddings(inputs)
    input = input.permute(1, 0, 2)
    h_0 = Variable(torch.zeros(2, input.size()[1], self.hidden_size).to(device))
    output, h_n = self.rnn(input, h_0)
    h_n = h_n.permute(1, 0, 2)
    h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
    logits = self.fc(h_n)

    return logits


# Uncomment to test VanillaRNN class
sampleRNN = VanillaRNN(10, 50, 1000, 300)
print(sampleRNN)