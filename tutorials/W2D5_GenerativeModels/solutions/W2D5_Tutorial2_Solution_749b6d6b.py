def ratio_disc(disc, x_real, x_fake):
  """Compute the density ratio between real distribution and fake distribution for `x`

  Args:
    disc: The discriminator
    x (ndarray): An array of shape (N,) that contains the samples to evaluate

  Returns:
    ndarray: The density ratios
  """

  # Put samples together
  x = torch.cat([x_real, x_fake])

  # Compute p / (p + q)
  p_over_pplusq = disc.classify(x)

  # Compute q / (p + q)
  q_over_pplusq = 1 - p_over_pplusq

  # Compute p / q
  p_over_q = p_over_pplusq / q_over_pplusq

  return p_over_q


# add event to airtable
atform.add_event('Coding Exercise 3.2: Estimating the density ratio by the discriminator')

disc = OptimalDisc()
gen = DummyGen()

x_real = get_data(1_000)
x_fake = gen.sample(1_000)

## Uncomment below to check your function
ratio = ratio_disc(disc, x_real, x_fake)
with plt.xkcd():
  plotting_ratio(x_real, x_fake, ratio)