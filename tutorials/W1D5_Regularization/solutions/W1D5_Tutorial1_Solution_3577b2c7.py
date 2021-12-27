def calculate_frobenius_norm(model):
  norm = 0.0
  # Sum the square of all parameters
  for param in model.parameters():
    norm += torch.sum(param**2)

  # Take a square root of the sum of squares of all the parameters
  norm = norm**0.5
  return norm


# add event to airtable
atform.add_event('Coding Exercise 1: Frobenius Norm')

# Seed added for reproducibility
set_seed(seed=SEED)

## uncomment below to test your code
net = nn.Linear(10, 1)
print(f'Frobenius norm of Single Linear Layer: {calculate_frobenius_norm(net)}')