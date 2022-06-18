def momentum_update(loss, params, grad_vel, lr=1e-3, beta=0.8):
  """
  Perform a momentum update over a collection of parameters given a loss and velocities

  Args:
    loss: Tensor
      A scalar tensor containing the loss through which gradient will be computed
    params: Iterable
      Collection of parameters with respect to which we compute gradients
    grad_vel: Iterable
      Collection containing the 'velocity' v_t for each parameter
    lr: Float
      Scalar specifying the learning rate or step-size for the update
    beta: Float
      Scalar 'momentum' parameter

  Returns:
    Nothing
  """
  # Clear up gradients as Pytorch automatically accumulates gradients from
  # successive backward calls
  zero_grad(params)
  # Compute gradients on given objective
  loss.backward()

  with torch.no_grad():
    for (par, vel) in zip(params, grad_vel):
      # Update 'velocity'
      vel.data = -lr * par.grad.data + beta * vel.data
      # Update parameters
      par.data += vel.data


# add event to airtable
atform.add_event('Coding Exercise 4: Implement momentum')

set_seed(seed=SEED)
model2 = MLP(in_dim=784, out_dim=10, hidden_dims=[])
print('\n The model2 parameters before the update are: \n')
print_params(model2)
loss = loss_fn(model2(X), y)
initial_vel = [torch.randn_like(p) for p in model2.parameters()]

## Uncomment below to test your function
momentum_update(loss, list(model2.parameters()), grad_vel=initial_vel, lr=1e-1, beta=0.9)
print('\n The model2 parameters after the update are: \n')
print_params(model2)