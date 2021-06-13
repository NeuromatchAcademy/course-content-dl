def train_svd_exercise(model, in_features, out_features, n_epochs, lr):
  """Training function

  Args:
    model (torch nn.Module): the neural network
    in_features (torch.Tensor): features (input) with shape `torch.Size([batch_size, input_dim])`
    out_features (torch.Tensor): targets (labels) with shape `torch.Size([batch_size, output_dim])`
    n_epochs (int): number of training epochs
    lr(float): learning rate

  Returns:
    np.ndarray: record (evolution) of losses
    np.ndarray: record (evolution) of singular values
  """

  assert in_features.shape[0] == out_features.shape[0]
  optimizer = optim.SGD(model.parameters(), lr=lr)
  criterion = nn.MSELoss()
  loss_record = []  # for recoding losses
  sv_record = []  # for recoding singular values

  for i in range(n_epochs):
    y_pred = model(in_features)  # forward pass
    loss = criterion(y_pred, out_features)  # calculating the loss
    optimizer.zero_grad()  # reset all the graph gradients to zero
    loss.backward()  # back propagation of the error
    optimizer.step()  # gradient step

    # calculating the W_tot by multiplying all layers' weights
    W_tot = model.layers[-1].weight.detach()  # starting from the last layer
    for i in range(2, len(model.layers)+1):
      W_tot = W_tot @ model.layers[-i].weight.detach()

    # performing the SVD!
    U, Σ, V = torch.svd(W_tot)

    loss_record.append(loss.item())
    sv_record.append(Σ.numpy())

  return np.array(loss_record), np.array(sv_record)