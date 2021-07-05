def momentum_update(loss, params, grad_vel, lr=1e-1, beta=0.8):
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

  for (par, vel) in zip(params, grad_vel):
    # Update 'velocity'
    vel.data = -lr * par.grad.data + beta * vel.data
    # Update parameters
    par.data += vel.data