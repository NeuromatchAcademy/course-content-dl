class LinearAutoEncoder(nn.Module):
  def __init__(self, x_dim, h_dim):
    """A Linear AutoEncoder

    Args:
      x_dim (int): input dimension
      h_dim (int): hidden dimension, bottleneck dimension, K
    """
    super().__init__()
    # encoder layer
    self.enc_lin = nn.Linear(x_dim, h_dim)
    # decoder layer
    self.dec_lin = nn.Linear(h_dim, x_dim)

  def encode(self, x):
    h = self.enc_lin(x)
    return h

  def decode(self, h):
    x_prime = self.dec_lin(h)
    return x_prime

  def forward(self, x):
    flat_x = x.view(x.size(0), -1)
    h = self.encode(flat_x)
    return self.decode(h).view(x.size())


# Pick your own K
K = 20
set_seed(seed=SEED)
## Uncomment to test your code
lin_ae = LinearAutoEncoder(my_dataset_dim, K)
lin_losses = train_autoencoder(lin_ae, my_dataset)
with plt.xkcd():
  plot_linear_ae(lin_losses)