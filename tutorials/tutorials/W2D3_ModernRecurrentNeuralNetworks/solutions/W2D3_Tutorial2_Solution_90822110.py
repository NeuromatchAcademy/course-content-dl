class AttentionModel(torch.nn.Module):
  def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
    super(AttentionModel, self).__init__()

    self.batch_size = batch_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embedding_length = embedding_length

    self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
    self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
    self.lstm = nn.LSTM(embedding_length, hidden_size)
    self.fc1 = nn.Linear(2*hidden_size, output_size)

  def attention_net(self, lstm_output, final_state):
    """
    lstm_output : shape: (num_seq, batch_size, hidden_size)
    final_state : shape: (1, batch_size, hidden_size)
    """
    # permute the output to get the shape (batch_size, num_seq, hidden_size)
    # Get the attention weights
    # use torch.bmm to compute the attention weights between each output and hast hidden state
    # pay attention to the tensor shapes, you may have to use squeeze and unsqueeze functions
    # softmax the attention weights
    # Get the new hidden state, use torch.bmm to get the weighted lstm output
    # pay attention to the tensor shapes, you may have to use squeeze and unsqueeze functions
    lstm_output = lstm_output.permute(1, 0, 2)
    hidden = final_state.squeeze(0)
    attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)  # expected shape: (batch_size, num_seq)
    soft_attn_weights = F.softmax(attn_weights, 1)
    new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

    return new_hidden_state

  def forward(self, input_sentences):

    input = self.word_embeddings(input_sentences)
    input = input.permute(1, 0, 2)

    h_0 = Variable(torch.zeros(1, input.shape[1], self.hidden_size).cuda())
    c_0 = Variable(torch.zeros(1, input.shape[1], self.hidden_size).cuda())

    output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
    attn_output = self.attention_net(output, final_hidden_state)
    final_output = torch.cat((attn_output, final_hidden_state[0]), 1)
    logits = self.fc1(final_output)

    return logits


# Uncomment to check AttentionModel class
attention_model = AttentionModel(32, 2, 16, 20, 200, TEXT.vocab.vectors)
print(attention_model)