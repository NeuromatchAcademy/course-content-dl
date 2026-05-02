def loss_fn(model, x, marginal_prob_std, eps=1e-3, device='cpu'):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
      Note, it takes two inputs in its forward function model(x, t)
      $s_\theta(x,t)$ in the equation
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel, takes `t` as input.
      $\sigma_t$ in the equation.
    eps: A tolerance value for numerical stability.
  """
  # Sample time uniformly in eps, 1
  random_t = torch.rand(x.shape[0], device=device) * (1. - eps) + eps
  # Find the noise std at the time `t`
  std = marginal_prob_std(random_t).to(device)
  # get normally distributed noise N(0, I)
  z = torch.randn_like(x).to(device)
  # compute the perturbed x = x + z * \sigma_t
  perturbed_x = x + z * std[:, None, None, None]
  # predict score with the model at (perturbed x, t)
  score = model(perturbed_x, random_t)
  # compute distance between the score and noise \| score * sigma_t + z \|_2^2
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
  ##############
  return loss