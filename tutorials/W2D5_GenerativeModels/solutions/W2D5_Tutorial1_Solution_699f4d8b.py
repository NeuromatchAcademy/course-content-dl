class ConvAutoEncoder(nn.Module):
  def __init__(self, x_dim, h_dim, n_filters=32, filter_size=5):
    """A Convolutional AutoEncoder

    Args:
      x_dim (tuple): input dimensions (channels, height, widths)
      h_dim (int): hidden dimension, bottleneck dimension, K
      n_filters (int): number of filters (number of output channels)
      filter_size (int): kernel size
    """
    super().__init__()
    channels, height, widths = x_dim

    # encoder input bias layer
    self.enc_bias = BiasLayer(x_dim)

    # first encoder conv2d layer
    self.enc_conv_1 = nn.Conv2d(channels, n_filters, filter_size)

    # output shape of the first encoder conv2d layer given x_dim input
    conv_1_shape = cout(x_dim, self.enc_conv_1)

    # second encoder conv2d layer
    self.enc_conv_2 = nn.Conv2d(n_filters, n_filters, filter_size)

    # output shape of the second encoder conv2d layer given conv_1_shape input
    conv_2_shape = cout(conv_1_shape, self.enc_conv_2)

    # The bottleneck is a dense layer, therefore we need a flattenning layer
    self.enc_flatten = nn.Flatten()

    # conv output shape is (depth, height, width), so the flatten size is:
    flat_after_conv = conv_2_shape[0] * conv_2_shape[1] * conv_2_shape[2]

    # encoder Linear layer
    self.enc_lin = nn.Linear(flat_after_conv, h_dim)

    # decoder Linear layer
    self.dec_lin = nn.Linear(h_dim, flat_after_conv)

    # unflatten data to (depth, height, width) shape
    self.dec_unflatten = nn.Unflatten(dim=-1, unflattened_size=conv_2_shape)

    # first "deconvolution" layer
    self.dec_deconv_1 = nn.ConvTranspose2d(n_filters, n_filters, filter_size)

    # second "deconvolution" layer
    self.dec_deconv_2 = nn.ConvTranspose2d(n_filters, channels, filter_size)

    # decoder output bias layer
    self.dec_bias = BiasLayer(x_dim)

  def encode(self, x):
    s = self.enc_bias(x)
    s = F.relu(self.enc_conv_1(s))
    s = F.relu(self.enc_conv_2(s))
    s = self.enc_flatten(s)
    h = self.enc_lin(s)
    return h

  def decode(self, h):
    s = F.relu(self.dec_lin(h))
    s = self.dec_unflatten(s)
    s = F.relu(self.dec_deconv_1(s))
    s = self.dec_deconv_2(s)
    x_prime = self.dec_bias(s)
    return x_prime

  def forward(self, x):
    return self.decode(self.encode(x))


K = 20
set_seed(seed=SEED)
## Uncomment to test your solution
conv_ae = ConvAutoEncoder(my_dataset_size, K)
assert conv_ae.encode(my_dataset[0][0].unsqueeze(0)).numel() == K, "Encoder output size should be K!"
conv_losses = train_autoencoder(conv_ae, my_dataset, device=DEVICE, seed=SEED)
with plt.xkcd():
  plot_conv_ae(lin_losses, conv_losses)