def l2_reg(model):
  """
  This function calculates the l2 norm of the all the tensors in the model

  Args:
    model: nn.module
      Neural network instance

  Returns:
    l2: float
      L2 norm of the all the tensors in the model
  """

  l2 = 0.0
  for param in model.parameters():
    l2 += torch.sum(torch.abs(param)**2)

  return l2

# Add event to airtable
atform.add_event('Coding Exercise 1.2: L2 Regularization')

set_seed(SEED)
## uncomment to test
net = nn.Linear(20, 20)
print(f"L2 norm of the model: {l2_reg(net)}")