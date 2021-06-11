
# Computing expression 2:

def dot_product():
###############################################
## TODO for students: create the b1 and b2 matrices
## from the second expression
  #raise NotImplementedError("Student exercise: fill in the missing code to complete the operation")
  b1 = torch.tensor([3,5,7])
  b2 = torch.tensor([2,4,8])
  product = torch.dot(b1,b2)
  return product

## TODO for students: compute the expression above and assign
## the result to a tensor named b

b = dot_product()
print(b)