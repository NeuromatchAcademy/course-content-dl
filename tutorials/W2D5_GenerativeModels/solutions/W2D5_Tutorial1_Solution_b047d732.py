class ConvAutoEncoder(nn.Module):
  def __init__(self, K, num_filters=32, filter_size=5):
    super(ConvAutoEncoder, self).__init__()

    # With padding=0, the number of pixels cut off from each image dimension
    # is filter_size // 2. Double it to get the amount of pixels lost in
    # width and height per Conv2D layer, or added back in per
    # ConvTranspose2D layer.
    filter_reduction = 2 * (filter_size // 2)

    # After passing input through two Conv2d layers, the shape will be
    # 'shape_after_conv'. This is also the shape that will go into the first
    # deconvolution layer in the decoder
    self.shape_after_conv = (num_filters,
                              my_dataset_size[1]-2*filter_reduction,
                              my_dataset_size[2]-2*filter_reduction)
    flat_size_after_conv = self.shape_after_conv[0] \
        * self.shape_after_conv[1] \
        * self.shape_after_conv[2]

    # Create encoder layers (BiasLayer, Conv2d, Conv2d, Flatten, Linear)
    self.enc_bias = BiasLayer(my_dataset_size)
    self.enc_conv_1 = nn.Conv2d(my_dataset_size[0], num_filters, filter_size)
    self.enc_conv_2 = nn.Conv2d(num_filters, num_filters, filter_size)
    self.enc_flatten = nn.Flatten()
    self.enc_lin = nn.Linear(flat_size_after_conv, K)

    # Create decoder layers (Linear, Unflatten(-1, self.shape_after_conv), ConvTranspose2d, ConvTranspose2d, BiasLayer)
    self.dec_lin = nn.Linear(K, flat_size_after_conv)
    self.dec_unflatten = nn.Unflatten(dim=-1, unflattened_size=self.shape_after_conv)
    self.dec_deconv_1 = nn.ConvTranspose2d(num_filters, num_filters, filter_size)
    self.dec_deconv_2 = nn.ConvTranspose2d(num_filters, my_dataset_size[0], filter_size)
    self.dec_bias = BiasLayer(my_dataset_size)

  def encode(self, x):
    # Your code here: encode batch of images (don't forget ReLUs!)
    s = self.enc_bias(x)
    s = F.relu(self.enc_conv_1(s))
    s = F.relu(self.enc_conv_2(s))
    s = self.enc_flatten(s)
    h = self.enc_lin(s)
    return h

  def decode(self, h):
    # Your code here: decode batch of h vectors (don't forget ReLUs!)
    s = F.relu(self.dec_lin(h))
    s = self.dec_unflatten(s)
    s = F.relu(self.dec_deconv_1(s))
    s = self.dec_deconv_2(s)
    x_prime = self.dec_bias(s)
    return x_prime

  def forward(self, x):
    return self.decode(self.encode(x))


K = 20
set_seed(2021)
# Uncomment to test your solution
conv_ae = ConvAutoEncoder(K=K)
assert conv_ae.encode(my_dataset[0][0].unsqueeze(0)).numel() == K, "Encoder output size should be K!"
conv_losses = train_autoencoder(conv_ae, my_dataset)
with plt.xkcd():
  plot_conv_ae(lin_losses, conv_losses)