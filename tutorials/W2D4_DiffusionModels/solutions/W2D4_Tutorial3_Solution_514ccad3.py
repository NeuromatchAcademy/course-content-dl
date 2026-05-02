def marginal_prob_std(t, Lambda, device='cpu'):
  """Compute the standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:
    t: A vector of time steps.
    Lambda: The $\lambda$ in our SDE.

  Returns:
    std : The standard deviation.
  """
  t = t.to(device)
  std = torch.sqrt((Lambda**(2 * t) - 1.) / 2. / np.log(Lambda))
  return std


def diffusion_coeff(t, Lambda, device='cpu'):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    Lambda: The $\lambda$ in our SDE.

  Returns:
    diff_coeff : The vector of diffusion coefficients.
  """
  diff_coeff = Lambda**t
  return diff_coeff.to(device)