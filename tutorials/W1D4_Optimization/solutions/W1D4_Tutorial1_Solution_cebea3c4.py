def rmsprop_update(loss, params, grad_sq, lr=1e-1, alpha=0.8):
  """Perform an RMSprop update on a collection of parameters

  Args:
    loss (tensor): A scalar tensor containing the loss whose gradient will be computed
    params (iterable): Collection of parameters with respect to which we compute gradients
    grad_sq (iterable): Moving average of squared gradients
    lr (float): Scalar specifying the learning rate or step-size for the update
    alpha (float): Moving average parameter
  """
  # Clear up gradients as Pytorch automatically accumulates gradients from
  # successive backward calls
  zero_grad(params)
  # Compute gradients on given objective
  loss.backward()

  for (par, gsq) in zip(params, grad_sq):
    # Update estimate of gradient variance
    gsq.data = alpha * gsq.data + (1-alpha) * par.grad**2
    # Update parameters
    par.data -=  lr * (par.grad / (1e-8 + gsq.data)**0.5)


set_seed(2021)
model = MLP(in_dim=784, out_dim=10, hidden_dims=[])
print('\n The model parameters before the update are: \n')
print_params(model)
loss = loss_fn(model(X), y).to(DEVICE)
grad_sq = [0.0001*i for i in list(model.parameters())]

## Uncomment below to test your function
rmsprop_update(loss, list(model.parameters()), grad_sq=grad_sq, lr=1e-2)
print('\n The model parameters after the update are: \n')
print_params(model)