def train(features, labels, model, loss_fun, optimizer, n_epochs):

  """Training function

  Args:
    features (torch.Tensor): features (input) with shape torch.Size([n_samples, 2])
    labels (torch.Tensor): labels (targets) with shape torch.Size([n_samples, 2])
    model (torch nn.Module): the neural network
    loss_fun (function): loss function
    optimizer(function): optimizer
    n_epochs (int): number of training epochs

  Returns:
    list: record (evolution) of losses
    list: record (evolution) of value of the first parameter
    list: record (evolution) of gradient of the first parameter
  """
  loss_record = []  # keeping recods of loss
  par_values = []  # keeping recods of first parameter
  par_grads = []  # keeping recods of gradient of first parameter

  # we use `tqdm` methods for progress bar
  epoch_range = trange(n_epochs, desc='loss: ', leave=True)
  for i in epoch_range:

    if loss_record:
      epoch_range.set_description("loss: {:.4f}".format(loss_record[-1]))
      epoch_range.refresh()  # to show immediately the update
      time.sleep(0.01)

    optimizer.zero_grad()  # Initialize gradients to 0
    predictions = model(features)  # Compute model prediction (output)
    loss = loss_fun(predictions, labels)  # Compute the loss
    loss.backward()  # Compute gradients (backward pass)
    optimizer.step()  # update parameters (optimizer takes a step)

    loss_record.append(loss.item())
    par_values.append(next(model.parameters())[0][0].item())
    par_grads.append(next(model.parameters()).grad[0][0].item())

  return loss_record, par_values, par_grads


epochs = 5000
losses, values, gradients = train(X, y,
                                  naive_model,
                                  cross_entropy_loss,
                                  sgd_optimizer,
                                  epochs)

with plt.xkcd():
  ex3_plot(epochs, losses, values, gradients)