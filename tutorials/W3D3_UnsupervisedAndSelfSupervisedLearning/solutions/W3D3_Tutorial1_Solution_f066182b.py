def plot_rsms(seed):
  """
  Helper function to plot Representational Similarity Matrices (RSMs)

  Args:
    seed: Integer
      The seed value for the dataset/network

  Returns:
    random_encoder: nn.module
      The encoder with mentioned parameters/hyperparameters
  """
  # Call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)

  # Initialize a core encoder network that will not get trained
  random_encoder = models.EncoderCore()

  # Try sorting by different latent dimensions
  sorting_latent = "shape"

  # Plot RSMs
  print("Plotting RSMs...")
  _ = models.plot_model_RSMs(
      encoders=[supervised_encoder, random_encoder],  # Pass both encoders
      dataset=dSprites_torchdataset,
      sampler=test_sampler,  # To see the representations on the held out test set
      titles=["Supervised network encoder RSM",
              "Random network encoder RSM"],  # Plot titles
      sorting_latent=sorting_latent,
      )

  return random_encoder



## Uncomment below to test your function
with plt.xkcd():
  random_encoder = plot_rsms(seed=SEED)