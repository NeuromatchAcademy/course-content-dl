def simple_operations(a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor):
  # multiplication of tensor a1 with tensor a2 and then add it with tensor a3
  answer = a1 @ a2 + a3
  return answer


# Computing expression 1:

# init our tensors
a1 = torch.tensor([[2, 4], [5, 7]])
a2 = torch.tensor([[1, 1], [2, 3]])
a3 = torch.tensor([[10, 10], [12, 1]])
## uncomment to test your function
A = simple_operations(a1, a2, a3)
print(A)