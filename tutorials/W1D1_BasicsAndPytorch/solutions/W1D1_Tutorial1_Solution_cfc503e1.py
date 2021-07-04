# Computing expression 1:
def simple_operations(a1):

  a2 = torch.tensor([[1, 1], [2, 3]])
  a3 = torch.tensor([[10, 10],[12, 1]])

  answer = a1 @ a2 + a3
  return answer


# init our tensors
a1 = torch.tensor([[2, 4], [5, 7]])
## Uncomment below to test your function
A = simple_operations(a1)
print(A)