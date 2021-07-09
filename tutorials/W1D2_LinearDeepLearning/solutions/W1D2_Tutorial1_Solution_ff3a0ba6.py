def train(features, labels, model, loss_fun, optimizer, n_epochs):

  """Training function

  Args:
    features (torch.Tensor): features (input) with shape torch.Size([n_samples, 1])
    labels (torch.Tensor): labels (targets) with shape torch.Size([n_samples, 1])
    model (torch nn.Module): the neural network
    loss_fun (function): loss function
    optimizer(function): optimizer
    n_epochs (int): number of training iterations

  Returns:
    list: record (evolution) of training losses
  """
  loss_record = []  # keeping recods of loss

  for i in range(n_epochs):
    optimizer.zero_grad()  # set gradients to 0
    predictions = model(features)  # Compute model prediction (output)
    loss = loss_fun(predictions, labels)  # Compute the loss
    loss.backward()  # Compute gradients (backward pass)
    optimizer.step()  # update parameters (optimizer takes a step)

    loss_record.append(loss.item())
  return loss_record


set_seed(seed=2021)
epochs = 1847 # Cauchy, Exercices d'analyse et de physique mathematique (1847)
losses = train(inputs, targets, wide_net, loss_function, sgd_optimizer, epochs)
with plt.xkcd():
  ex3_plot(wide_net, inputs, targets, epochs, losses)