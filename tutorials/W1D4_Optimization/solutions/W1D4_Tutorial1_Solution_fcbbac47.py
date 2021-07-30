def rmsprop_update(loss, params, grad_sq, lr=1e-3, alpha=0.8, epsilon=1e-8):
  """Perform an RMSprop update on a collection of parameters

  Args:
    loss (tensor): A scalar tensor containing the loss whose gradient will be computed
    params (iterable): Collection of parameters with respect to which we compute gradients
    grad_sq (iterable): Moving average of squared gradients
    lr (float): Scalar specifying the learning rate or step-size for the update
    alpha (float): Moving average parameter
    epsilon (float): for numerical estability
  """
  # Clear up gradients as Pytorch automatically accumulates gradients from
  # successive backward calls
  zero_grad(params)
  # Compute gradients on given objective
  loss.backward()

  with torch.no_grad():
    for (par, gsq) in zip(params, grad_sq):
      # Update estimate of gradient variance
      gsq.data = alpha * gsq.data + (1 - alpha) * par.grad**2
      # Update parameters
      par.data -=  lr * (par.grad / (epsilon + gsq.data)**0.5)


set_seed(seed=SEED)
model3 = MLP(in_dim=784, out_dim=10, hidden_dims=[])
print('\n The model3 parameters before the update are: \n')
print_params(model3)
loss = loss_fn(model3(X), y)
# Intialize the moving average of squared gradients
grad_sq = [1e-6*i for i in list(model3.parameters())]

## Uncomment below to test your function
rmsprop_update(loss, list(model3.parameters()), grad_sq=grad_sq, lr=1e-3)
print('\n The model3 parameters after the update are: \n')
print_params(model3)