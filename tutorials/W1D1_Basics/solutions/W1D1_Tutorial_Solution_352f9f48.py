
# Computing expression 1:

# init our tensors
a1 = torch.tensor([[2, 4], [5, 7]])

def simple_operations(a1):
################################################
## TODO for students: create the a2 and a3 matrices
## from the first expression
  #raise NotImplementedError("Student exercise: fill in the missing code to complete the operation")
  a2 = torch.tensor([[1,1], [2,3]])
  a3 = torch.tensor([[10,10],[12,1]])

  answer = a1@a2+a3
  return answer

## TODO for students: compute the expression above and assign
## the result to a tensor named A

A = simple_operations(a1)

print(A)