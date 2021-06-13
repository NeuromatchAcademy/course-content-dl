class ShallowNarrowExercise:
  """
  Shallow and narrow (one neuron per layer) linear neural network
  """
  def __init__(self, init_ws: list):
    """
    init_ws: initial weights as a list
    """
    assert isinstance(init_ws, list)
    assert len(init_ws) == 2
    self.w1 = init_ws[0]
    self.w2 = init_ws[1]


  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    The forward pass through netwrok y = x * w1 * w2
    """
    y = x * self.w1 * self.w2
    return y


  def dloss_dw(self, x: np.ndarray, y_t: np.ndarray):
    """
    partial derivative of loss with respect to weights
    """
    assert x.shape == y_t.shape
    dloss_dw1 = - (2 * self.w2 * x * (y_t - self.w1 * self.w2 * x)).mean()
    dloss_dw2 = - (2 * self.w1 * x * (y_t - self.w1 * self.w2 * x)).mean()
    return dloss_dw1, dloss_dw2


  def train(self, x: np.ndarray, y_t: np.ndarray, η: float, n_ep: int):
    """
    Gradient descent algorithm
    """
    assert x.shape == y_t.shape

    loss_records = np.empty(n_ep)  # pre allocation of loss records
    weight_records = np.empty((n_ep, 2))  # pre allocation of weight records

    for i in range(n_ep):
      y_p = self.forward(x)
      loss_records[i] = loss(y_p, y_t)
      dloss_dw1, dloss_dw2 = self.dloss_dw(x, y_t)
      self.w1 -= η * dloss_dw1
      self.w2 -= η * dloss_dw2
      weight_records[i] = [self.w1, self.w2]

    return loss_records, weight_records


def loss(y_p: np.ndarray, y_t: np.ndarray):
  """
  Mean squared error
  """
  assert y_p.shape == y_t.shape
  mse = ((y_t - y_p)**2).mean()
  return mse


n_epochs = 211
learning_rate = 0.02
initial_weights = [1.4, -1.6]
x_train, y_train = gen_samples(n=73, a=2.0, σ=0.2)
x_eval = np.linspace(0.0, 1.0, 37, endpoint=True)

sn_model = ShallowNarrowExercise(initial_weights)
loss_log, weight_log = sn_model.train(x_train, y_train, learning_rate, n_epochs)
y_eval = sn_model.forward(x_eval)
with plt.xkcd():
  plot_x_y_(x_train, y_train, x_eval, y_eval, loss_log, weight_log)