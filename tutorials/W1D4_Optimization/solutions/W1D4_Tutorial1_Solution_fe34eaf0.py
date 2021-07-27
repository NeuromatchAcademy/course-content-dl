def momentum_update(loss, params, grad_vel, lr=1e-3, beta=0.8):
  """Perform a momentum update over a collection of parameters given a loss and 'velocities'

  Args:
    loss (tensor): A scalar tensor containing the loss whose gradient will be computed
    params (iterable): Collection of parameters with respect to which we compute gradients
    grad_vel (iterable): Collection containing the 'velocity' v_t for each parameter
    lr (float): Scalar specifying the learning rate or step-size for the update
    beta (float): Scalar 'momentum' parameter
  """
  # Clear up gradients as Pytorch automatically accumulates gradients from
  # successive backward calls
  zero_grad(params)
  # Compute gradients on given objective
  loss.backward()

  with torch.no_grad():
    for (par, vel) in zip(params, grad_vel):
      # Update 'velocity'
      vel = -lr * par.grad + beta * vel
      # Update parameters
      par += vel


set_seed(2021)
model = MLP(in_dim=784, out_dim=10, hidden_dims=[])
print('\n The model parameters before the update are: \n')
print_params(model)
loss = loss_fn(model(X), y)
initial_vel = [torch.randn_like(p) for p in model.parameters()]

## Uncomment below to test your function
momentum_update(loss, list(model.parameters()), grad_vel=initial_vel, lr=1e-1)
print('\n The model parameters after the update are: \n')
print_params(model)