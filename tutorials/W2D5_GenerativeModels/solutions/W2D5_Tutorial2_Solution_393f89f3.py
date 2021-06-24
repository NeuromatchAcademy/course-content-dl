def loss_disc(disc, x_real, x_fake):
  """Compute the discriminator loss for `x_real` and `x_fake` given `disc`

  Args:
    disc: The discriminator
    x_real (ndarray): An array of shape (N,) that contains the real samples
    x_fake (ndarray): An array of shape (N,) that contains the fake samples

  Returns:
    ndarray: The discriminator loss
  """

  # Loss for real data
  label_real = 1
  loss_real = label_real * torch.log(disc.classify(x_real))

  # Loss for fake data
  label_fake = 0
  loss_fake = (1 - label_fake) * torch.log(1 - disc.classify(x_fake))

  return torch.cat([loss_real, loss_fake])


disc = DummyDisc()
gen = DummyGen()

x_real = get_data()
x_fake = gen.sample()

## Uncomment to check your function
ld = loss_disc(disc, x_real, x_fake)
with plt.xkcd():
  plotting_ld(ld)