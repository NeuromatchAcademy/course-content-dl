def generate_images(autoencoder, K, n_images=1):
  """Generate n_images 'new' images from the decoder part of the given
  autoencoder.

  returns (n_images, channels, height, width) tensor of images
  """
  # Concatenate tuples to get (n_images, channels, height, width)
  output_shape = (n_images,) + my_dataset_size
  with torch.no_grad():
    # sample z, pass through autoencoder.decode(), and reshape output.
    z = torch.randn(n_images, K)
    x = autoencoder.decode(z).reshape(output_shape)

    return x


K = 20
set_seed(seed=SEED)
## Uncomment to run it
images = generate_images(conv_ae, K, n_images=25)
with plt.xkcd():
  plot_images(images)