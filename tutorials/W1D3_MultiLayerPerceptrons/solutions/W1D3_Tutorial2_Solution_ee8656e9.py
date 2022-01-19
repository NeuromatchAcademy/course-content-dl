def run_depth_optimizer(max_par_count, max_hidden_layer, device):
  """
  Simulate Depth Optimizer

  Args:
    max_par_count: int
      Maximum number of hidden units in layer of depth optimizer
    max_hidden_layer: int
      Maximum number of hidden layers within depth optimizer
    device: string
      CUDA/GPU if available, CPU otherwise

  Returns:
    hidden_layers: int
      Number of hidden layers in depth optimizer
    test_scores: list
      Log of test scores
  """

  def count_parameters(model):
    """
    Function to count model parameters

    Args:
      model: instance of Net class
        MLP instance

    Returns:
      par_count: int
        Number of parameters in network
    """
    par_count = 0
    for p in model.parameters():
      if p.requires_grad:
        par_count += p.numel()
    return par_count

  # Number of hidden layers to try
  hidden_layers = range(1, max_hidden_layer+1)

  # Test test score list
  test_scores = []

  for hidden_layer in hidden_layers:
    # Initialize the hidden units in each hidden layer to be 1
    hidden_units = np.ones(hidden_layer, dtype=np.int)

    # Define the the with hidden units equal to 1
    wide_net = Net('ReLU()', X_train.shape[1], hidden_units, K).to(device)
    par_count = count_parameters(wide_net)

    # Increment hidden_units and repeat until the par_count reaches the desired count
    while par_count < max_par_count:
      hidden_units += 1
      wide_net = Net('ReLU()', X_train.shape[1], hidden_units, K).to(device)
      par_count = count_parameters(wide_net)

    # Train it
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(wide_net.parameters(), lr=1e-3)
    _, test_acc = train_test_classification(wide_net, criterion, optimizer,
                                            train_loader, test_loader,
                                            num_epochs=100, device=device)
    test_scores += [test_acc]

  return hidden_layers, test_scores


# Add event to airtable
atform.add_event('Coding Exercise 1: Wide vs. Deep ')


set_seed(seed=SEED)
max_par_count = 100
max_hidden_layer = 5
## Uncomment below to test your function
hidden_layers, test_scores = run_depth_optimizer(max_par_count, max_hidden_layer, DEVICE)
with plt.xkcd():
  plt.xlabel('# of hidden layers')
  plt.ylabel('Test accuracy')
  plt.plot(hidden_layers, test_scores)
  plt.show()