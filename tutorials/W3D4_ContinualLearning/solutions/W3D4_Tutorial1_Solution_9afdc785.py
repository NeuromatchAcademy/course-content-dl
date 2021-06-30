def train_model():
  """Train a model with a CL algorithm of your choice on a chosen CL benchmark.
     The benchmark will have T tasks.

  Returns:
    result_matrix: TxT matrix of accuracies. Each (i,j) element is the accuracy
        on task j after training on task i
  """

  model = Net().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  criterion = nn.CrossEntropyLoss()

  benchmark = None  # define your CL benchmark

  # train the model on the benchmark with the strategy and get the result
  result_matrix = train_multihead(past_examples_percentage=0.5, epochs=10)

  return np.array(result_matrix)