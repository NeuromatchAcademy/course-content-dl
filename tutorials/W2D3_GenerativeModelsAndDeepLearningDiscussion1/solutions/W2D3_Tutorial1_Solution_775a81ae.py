def generate_images(autoencoder, K, n_images=1):
  """
  Generate n_images 'new' images from the decoder part of the given
  autoencoder.

  Args:
    autoencoder: nn.module
      Autoencoder model
    K: int
      Bottleneck dimension
    n_images: int
      Number of images

  Returns:
    x: torch.tensor
      (n_images, channels, height, width) tensor of images
  """
  # Concatenate tuples to get (n_images, channels, height, width)
  output_shape = (n_images,) + data_shape
  with torch.no_grad():
    # Sample z from a unit gaussian, pass through autoencoder.decode()
    z = torch.randn(n_images, K)
    x = autoencoder.decode(z)

    return x.reshape(output_shape)



set_seed(seed=SEED)
## Uncomment to test your solution
images = generate_images(trained_conv_AE, K, n_images=9)
plot_images(images, plt_title='Images Generated from the Conv-AE')
images = generate_images(trained_conv_VarAE, K_VAE, n_images=9)
plot_images(images, plt_title='Images Generated from a Conv-Variational-AE')