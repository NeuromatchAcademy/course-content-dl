def rsample(phi, n_samples):
  """Sample z ~ q(z;phi)
  Ouput z is size [b,n_samples,K] given phi with shape [b,K+1]. The first K
  entries of each row of phi are the mean of q, and phi[:,-1] is the log
  standard deviation
  """
  b, kplus1 = phi.size()
  k = kplus1 - 1
  mu, sig = phi[:, :-1], phi[:, -1].exp()
  eps = torch.randn(b, n_samples, k, device=phi.device)
  return eps * sig.view(b, 1, 1) + mu.view(b, 1, k)


# # Uncomment below to test your code
phi = torch.randn(4, 3, device=DEVICE)
set_seed(2021)
zs = rsample(phi, 100)
assert zs.size() == (4, 100, 2), "rsample size is incorrect!"
assert zs.device == phi.device, "rsample device doesn't match phi device!"
zs = zs.cpu()
with plt.xkcd():
  plot_phi(phi)