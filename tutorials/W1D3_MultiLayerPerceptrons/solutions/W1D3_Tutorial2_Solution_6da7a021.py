def run_lazy_training(num_time_steps, num_select_weights, step_epoch):

  # Define a wide MLP
  net = Net('ReLU()', X_train.shape[1], [1000], K).to(dev)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=1e-2)

  # let's save only couple of parameters at each time step
  weights = torch.zeros(num_time_steps, num_select_weights)
  for i in range(num_time_steps):
    _, _ = train_test_classification(net, criterion, optimizer, train_loader,
                                    test_loader, num_epochs=step_epoch, verbose=False)

    # let's pratice some tensor navigations!
    # access the first layer weights
    # and index the first column
    # and slice upto num_select_weights paramaeters
    weights[i] = net.layers[0].weight[:num_select_weights, 0]

  return weights

### Uncomment below to test your function
num_select_weights = 10
num_time_steps = 5
step_epoch = 50
weights = run_lazy_training(num_time_steps, num_select_weights, step_epoch)
with plt.xkcd():
  for k in range(num_select_weights):
    weight = weights[:, k].detach()
    epochs = range(1, 1+num_time_steps*step_epoch, step_epoch)
    plt.plot(epochs, weight, label='weight #%d'%k)

  plt.xlabel('epochs')
  plt.legend()
  plt.show()