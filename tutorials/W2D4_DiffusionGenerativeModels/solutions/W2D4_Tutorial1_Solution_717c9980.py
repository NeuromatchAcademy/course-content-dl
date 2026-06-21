def loss_fn(model, x_0, sigma_t_fun, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model:        Time-dependent score model; takes (x, t) and returns predicted score
    x_0:          Mini-batch of clean training data, shape (batch_size, 2)
    sigma_t_fun:  Function returning sigma_t given t, the std of p(x_t | x_0)
    eps:          Lower bound for t sampling; avoids sigma_t -> 0 blowing up the loss
  """
  # random_t shape: (batch_size,) — one time value per sample, sampled continuously from [eps, 1]
  # NOTE: unlike reverse diffusion which uses discrete steps 0..nsteps, training samples t
  # continuously so the model learns the score for all t in [eps, 1] simultaneously.
  random_t = torch.rand(x_0.shape[0], device=x_0.device) * (1. - eps) + eps

  # z shape: (batch_size, 2) — Gaussian noise, same shape as x_0
  z = torch.randn_like(x_0)

  # std shape: (batch_size,) — per-sample sigma_t value
  # .unsqueeze(1) → (batch_size, 1) so it broadcasts correctly against z (batch_size, 2)
  std = sigma_t_fun(random_t)

  # x_t shape: (batch_size, 2) — noisy data at time t: x_t = x_0 + sigma_t * z
  x_t = x_0 + z * std.unsqueeze(1)

  # Predict score s_theta(x_t, t), shape: (batch_size, 2)
  score = model(x_t, random_t)

  # Compute loss: ||sigma_t * s_theta(x_t, t) + z||^2, averaged over batch
  loss = torch.mean(torch.sum((score * std.unsqueeze(1) + z)**2, dim=1))
  return loss