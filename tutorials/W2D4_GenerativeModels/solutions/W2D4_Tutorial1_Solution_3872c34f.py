class LinearAutoEncoder(nn.Module):
  """
  Linear Autoencoder
  """

  def __init__(self, x_dim, h_dim):
    """
    A Linear AutoEncoder

    Args:
      x_dim: int
        Input dimension
      h_dim: int
        Hidden dimension, bottleneck dimension, K

    Returns:
      Nothing
    """
    super().__init__()
    # Encoder layer (a linear mapping from x_dim to K)
    self.enc_lin = nn.Linear(x_dim, h_dim)
    # Decoder layer (a linear mapping from K to x_dim)
    self.dec_lin = nn.Linear(h_dim, x_dim)

  def encode(self, x):
    """
    Encoder function

    Args:
      x: torch.tensor
        Input features

    Returns:
      x: torch.tensor
        Encoded output
    """
    h = self.enc_lin(x)
    return h

  def decode(self, h):
    """
    Decoder function

    Args:
      h: torch.tensor
        Encoded output

    Returns:
      x_prime: torch.tensor
        Decoded output
    """
    x_prime = self.dec_lin(h)
    return x_prime

  def forward(self, x):
    """
    Forward pass

    Args:
      x: torch.tensor
        Input data

    Returns:
      Decoded output
    """
    flat_x = x.view(x.size(0), -1)
    h = self.encode(flat_x)
    return self.decode(h).view(x.size())



# Pick your own K
K = 20
set_seed(seed=SEED)
## Uncomment to test your code
lin_ae = LinearAutoEncoder(data_size, K)
lin_losses = train_autoencoder(lin_ae, train_set, device=DEVICE, seed=SEED)
with plt.xkcd():
  plot_linear_ae(lin_losses)