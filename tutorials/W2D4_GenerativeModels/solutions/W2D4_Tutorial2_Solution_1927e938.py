def loss_fn(model, x, sigma_t_fun, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
      it takes x, t as arguments.
    x: A mini-batch of training data.
    sigma_t_fun: A function that gives the standard deviation of the conditional dist.
        p(x_t | x_0)
    eps: A tolerance value for numerical stability, sample t uniformly from [eps, 1.0]
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = sigma_t_fun(random_t, )
  perturbed_x = x + z * std[:, None]
  # use the model to predict score at x_t and t
  score = model(perturbed_x, random_t)
  # implement the loss \|\sigma_t s_\theta(x+\sigma_t z, t) + z\|^2
  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=(1)))
  return loss