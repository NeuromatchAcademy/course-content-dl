class ShallowNarrowExercise:
  """
  Shallow and narrow (one neuron per layer) linear neural network
  """

  def __init__(self, init_weights):
    """
    Initialize parameters of ShallowNarrow Net

    Args:
      init_weights: list
        Initial weights

    Returns:
      Nothing
    """
    assert isinstance(init_weights, (list, np.ndarray, tuple))
    assert len(init_weights) == 2
    self.w1 = init_weights[0]
    self.w2 = init_weights[1]


  def forward(self, x):
    """
    The forward pass through netwrok y = x * w1 * w2

    Args:
      x: np.ndarray
        Features (inputs) to neural net

    Returns:
      y: np.ndarray
        Neural network output (predictions)
    """
    y = x * self.w1 * self.w2
    return y


  def dloss_dw(self, x, y_true):
    """
    Gradient of loss with respect to weights

    Args:
      x: np.ndarray
        Features (inputs) to neural net
      y_true: np.ndarray
        True labels

    Returns:
      dloss_dw1: float
        Mean gradient of loss with respect to w1
      dloss_dw2: float
        Mean gradient of loss with respect to w2
    """
    assert x.shape == y_true.shape
    dloss_dw1 = - (2 * self.w2 * x * (y_true - self.w1 * self.w2 * x)).mean()
    dloss_dw2 = - (2 * self.w1 * x * (y_true - self.w1 * self.w2 * x)).mean()
    return dloss_dw1, dloss_dw2


  def train(self, x, y_true, lr, n_ep):
    """
    Training with Gradient descent algorithm

    Args:
      x: np.ndarray
        Features (inputs) to neural net
      y_true: np.ndarray
        True labels
      lr: float
        Learning rate
      n_ep: int
        Number of epochs (training iterations)

    Returns:
      loss_records: list
        Training loss records
      weight_records: list
        Training weight records (evolution of weights)
    """
    assert x.shape == y_true.shape

    loss_records = np.empty(n_ep)  # Pre allocation of loss records
    weight_records = np.empty((n_ep, 2))  # Pre allocation of weight records

    for i in range(n_ep):
      y_prediction = self.forward(x)
      loss_records[i] = loss(y_prediction, y_true)
      dloss_dw1, dloss_dw2 = self.dloss_dw(x, y_true)
      self.w1 -= lr * dloss_dw1
      self.w2 -= lr * dloss_dw2
      weight_records[i] = [self.w1, self.w2]

    return loss_records, weight_records


def loss(y_prediction, y_true):
  """
  Mean squared error

  Args:
    y_prediction: np.ndarray
      Model output (prediction)
    y_true: np.ndarray
      True label

  Returns:
    mse: np.ndarray
      Mean squared error loss
  """
  assert y_prediction.shape == y_true.shape
  mse = ((y_true - y_prediction)**2).mean()
  return mse

# Add event to airtable
atform.add_event('Coding Exercise 1.1: Implement simple narrow LNN')


set_seed(seed=SEED)
n_epochs = 211
learning_rate = 0.02
initial_weights = [1.4, -1.6]
x_train, y_train = gen_samples(n=73, a=2.0, sigma=0.2)
x_eval = np.linspace(0.0, 1.0, 37, endpoint=True)
## Uncomment to run
sn_model = ShallowNarrowExercise(initial_weights)
loss_log, weight_log = sn_model.train(x_train, y_train, learning_rate, n_epochs)
y_eval = sn_model.forward(x_eval)
with plt.xkcd():
  plot_x_y_(x_train, y_train, x_eval, y_eval, loss_log, weight_log)