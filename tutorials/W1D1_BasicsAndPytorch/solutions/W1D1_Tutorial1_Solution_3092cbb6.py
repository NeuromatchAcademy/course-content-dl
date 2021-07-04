def simpleFunGPU():

  x = torch.rand(10000, 10000).to("cuda")
  y = torch.rand_like(x).to("cuda")
  z = 2*torch.ones(10000, 10000).to("cuda")

  x = x * y
  x = x @ z


## Implement the function above and uncomment the following lines to test your code
timeFun(simpleFun, iterations=1)
timeFun(simpleFunGPU, iterations=1)