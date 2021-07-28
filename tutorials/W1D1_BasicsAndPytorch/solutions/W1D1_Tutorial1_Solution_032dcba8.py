def simpleFun(dim, device):
  """
  Args:
    dim: integer
    device: "cpu" or "cuda"
  Returns:
    Nothing.
  """
  x = torch.rand(dim, dim).to(device)
  y = torch.rand_like(x).to(device)
  z = 2*torch.ones(dim, dim).to(device)

  x = x * y
  x = x @ z


  del x
  del y
  del z


## TODO: Implement the function above and uncomment the following lines to test your code
timeFun(f=simpleFun, dim=dim, iterations=iterations)
timeFun(f=simpleFun, dim=dim, iterations=iterations, device=DEVICE)