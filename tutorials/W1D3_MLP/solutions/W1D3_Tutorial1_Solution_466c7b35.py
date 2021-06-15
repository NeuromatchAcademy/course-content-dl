
def approximate_function(x_train, y_train):

    # Number of relus
    n_relus = x_train.shape[0] - 1

    # x axis points (more than x train)
    x = torch.linspace(torch.min(x_train), torch.max(x_train), 1000)

    ## COMPUTE RELU ACTIVATIONS

    # First determine what bias terms should be for each of 9 ReLUs
    b = -x_train[:9]

    # Compute ReLU activations for each point along the x axis (x)
    relu_acts = torch.zeros((n_relus, x.shape[0]))

    for i_relu in range(n_relus):
      relu_acts[i_relu, :] = torch.relu(x + b[i_relu])


    ## COMBINE RELU ACTIVATIONS

    # Set up weights for weighted sum of ReLUs
    combination_weights = torch.zeros((n_relus, ))

    # Figure out weights on each ReLU
    prev_slope = 0
    for i in range(n_relus):
      delta_x = x_train[i+1] - x_train[i]
      slope = (y_train[i+1] - y_train[i]) / delta_x
      combination_weights[i] = slope - prev_slope
      prev_slope = slope

    # Get output of weighted sum of ReLU activations for every point along x axis
    y_hat = combination_weights @ relu_acts

    return y_hat, relu_acts, x

# Make training data from sine function
N_train = 10
x_train = torch.linspace(0, 2*np.pi, N_train).view(-1, 1)
y_train = torch.sin(x_train)


### uncomment the lines below to test your function approximation
y_hat, relu_acts, x = approximate_function(x_train, y_train)
with plt.xkcd():
  plot_function_approximation(x, relu_acts, y_hat)