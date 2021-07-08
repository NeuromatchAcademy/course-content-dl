class VanillaRNN(nn.Module):
  def __init__(self, layers, output_size, hidden_size, vocab_size, embed_size,
               device):
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
    input = self.embeddings(inputs)
    input = input.permute(1, 0, 2)
    h_0 = Variable(torch.zeros(2, input.size()[1],
                               self.hidden_size).to(self.device))
    output, h_n = self.rnn(input, h_0)
    h_n = h_n.permute(1, 0, 2)
    h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
    logits = self.fc(h_n)

    return logits


## Uncomment to test VanillaRNN class
sampleRNN = VanillaRNN(2, 10, 50, 1000, 300, DEVICE)
print(sampleRNN)