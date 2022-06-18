def l1_reg(model):
  """
  This function calculates the l1 norm of the all the tensors in the model

  Args:
    model: nn.module
      Neural network instance

  Returns:
    l1: float
      L1 norm of the all the tensors in the model
  """
  l1 = 0.0

  for param in model.parameters():
    l1 += torch.sum(torch.abs(param))

  return l1

# Add event to airtable
atform.add_event('Coding Exercise 1.1: L1 Regularization')

set_seed(seed=SEED)
## uncomment to test
net = nn.Linear(20, 20)
print(f"L1 norm of the model: {l1_reg(net)}")