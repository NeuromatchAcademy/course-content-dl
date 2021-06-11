def simpleFun():
  x = torch.rand(10000,10000)
  y = torch.rand_like(x)
  z = 2*torch.ones(10000,10000)

  x = x * y
  x = x @ z


def simpleFunGPU():
  x = torch.rand(10000,10000).to("cuda")
  y = torch.rand_like(x).to("cuda")
  z = 2*torch.ones(10000,10000).to("cuda")

  x = x * y
  x = x @ z

timeFun(simpleFun, iterations = 1)
timeFun(simpleFunGPU, iterations = 1)