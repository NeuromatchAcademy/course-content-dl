def dot_product(b1: torch.Tensor, b2: torch.Tensor):
  # Use torch.dot() to compute the dot product of two tensors
  product = torch.dot(b1, b2)
  return product

# add timing to airtable
atform.add_event('Coding Exercise 2.2 : Simple tensor operations-dot_product')


# Computing expression 2:
b1 = torch.tensor([3, 5, 7])
b2 = torch.tensor([2, 4, 8])
## Uncomment to test your function
b = dot_product(b1, b2)
print(b)