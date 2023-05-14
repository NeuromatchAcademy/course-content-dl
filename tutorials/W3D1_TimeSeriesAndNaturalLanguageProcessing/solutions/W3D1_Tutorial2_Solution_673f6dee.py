class TextCNN(nn.Module):
  def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                **kwargs):
    super(TextCNN, self).__init__(**kwargs)
    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.fc = nn.Linear(sum(num_channels), 2)
    self.pool = nn.AdaptiveMaxPool1d(1)
    self.relu = nn.ReLU()
    self.convs = nn.ModuleList()
    # This for loop adds the Conv1D layers to your network
    for c, k in zip(num_channels, kernel_sizes):
      # Conv1d(in_channels, out_channels, kernel_size)
      self.convs.append(nn.Conv1d(embed_size, c, k))

  def forward(self, inputs):
    embeddings = self.embedding(inputs)
    embeddings = embeddings.permute(0, 2, 1)
    # Concatenating the average-pooled outputs
    encoding = torch.cat([
        torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
        for conv in self.convs], dim=1)

    outputs = self.fc(encoding)
    return outputs


## Uncomment to test
sampleCNN = TextCNN(1000, 300, [1, 2, 3], [10, 20, 30])
print(sampleCNN)

# add event to airtable
atform.add_event('Coding Exercise 1: Implement a 1D CNN')