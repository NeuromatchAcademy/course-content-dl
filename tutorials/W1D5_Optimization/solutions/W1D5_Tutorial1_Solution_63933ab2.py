def gradient_update(loss, params, lr=1e-3):
  """
  Perform a gradient descent update on a given loss over a collection of parameters

  Args:
    loss: Tensor
      A scalar tensor containing the loss through which the gradient will be computed
    params: List of iterables
      Collection of parameters with respect to which we compute gradients
    lr: Float
      Scalar specifying the learning rate or step-size for the update

  Returns:
    Nothing
  """
  # Clear up gradients as Pytorch automatically accumulates gradients from
  # successive backward calls
  zero_grad(params)

  # Compute gradients on given objective
  loss.backward()

  with torch.no_grad():
    for par in params:
      #################################################
      ## TODO for students: update the value of the parameter ##
      # raise NotImplementedError("Student exercise: implement gradient update")
      #################################################
      # Here we work with the 'data' attribute of the parameter rather than the
      # parameter itself.
      # Hence - use the learning rate and the parameter's .grad.data attribute to perform an update
      par.data -= lr * par.grad.data


# add event to airtable
atform.add_event('Coding Exercise 3: Implement gradient descent')


set_seed(seed=SEED)
model1 = MLP(in_dim=784, out_dim=10, hidden_dims=[])
print('\n The model1 parameters before the update are: \n')
print_params(model1)
loss = loss_fn(model1(X), y)

## Uncomment below to test your function
gradient_update(loss, list(model1.parameters()), lr=1e-1)
print('\n The model1 parameters after the update are: \n')
print_params(model1)