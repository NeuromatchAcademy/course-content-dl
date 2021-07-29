
#add event to airtable
atform.add_event('Coding Exercise 2.1: Computational Graph ')

class SimpleGraph:
  def __init__(self, w, b):
    """Initializing the SimpleGraph

    Args:
      w (float): initial value for weight
      b (float): initial value for bias
    """
    assert isinstance(w, float)
    assert isinstance(b, float)
    self.w = torch.tensor([w], requires_grad=True)
    self.b = torch.tensor([b], requires_grad=True)

  def forward(self, x):
    """Forward pass

    Args:
      x (torch.Tensor): 1D tensor of features

    Returns:
      torch.Tensor: model predictions
    """
    assert isinstance(x, torch.Tensor)
    prediction = torch.tanh(x * self.w + self.b)
    return prediction


def sq_loss(y_true, y_prediction):
  """L2 loss function

  Args:
    y_true (torch.Tensor): 1D tensor of target labels
    y_prediction (torch.Tensor): 1D tensor of predictions

  Returns:
    torch.Tensor: L2-loss (squared error)
  """
  assert isinstance(y_true, torch.Tensor)
  assert isinstance(y_prediction, torch.Tensor)
  loss = (y_true - y_prediction)**2
  return loss



feature = torch.tensor([1])  # input tensor
target = torch.tensor([7])  # target tensor
## Uncomment to run
simple_graph = SimpleGraph(-0.5, 0.5)
print(f"initial weight = {simple_graph.w.item()}, "
      f"\ninitial bias = {simple_graph.b.item()}")
prediction = simple_graph.forward(feature)
square_loss = sq_loss(target, prediction)
print(f"for x={feature.item()} and y={target.item()}, "
      f"prediction={prediction.item()}, and L2 Loss = {square_loss.item()}")