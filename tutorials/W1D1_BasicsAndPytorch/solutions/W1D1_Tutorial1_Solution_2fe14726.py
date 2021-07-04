# Computing expression 2:
def dot_product():

  b1 = torch.tensor([3, 5, 7])
  b2 = torch.tensor([2, 4, 8])
  product = torch.dot(b1, b2)
  return product


## Uncomment below to test your function
b = dot_product()
print(b)