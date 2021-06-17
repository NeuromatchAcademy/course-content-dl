def l1_reg(model):
  """
    Inputs: Pytorch model
    This function calculates the l1 norm of the all the tensors in the model
  """
  l1 = 0.0

  for param in model.parameters():
    l1 += torch.sum(torch.abs(param))

  return l1


# # uncomment to test
net = nn.Linear(20, 20)
print(f'L1 norm of the model: {l1_reg(net)}')