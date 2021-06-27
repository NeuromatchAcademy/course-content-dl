def plot_rsms(seed):
  # call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)

  # initialize a core encoder network that will not get trained
  untrained_encoder = models.EncoderCore()

  # EXERCISE: Try sorting by different latent dimensions
  sorting_latent = "shape"

  # EXERCISE: Plot RSMs
  print("Plotting RSMs...")
  _ = models.plot_model_RSMs(
      encoders=[supervised_encoder, untrained_encoder],  # we pass both encoders
      dataset=dSprites_torchdataset,
      sampler=test_sampler,  # we want to see the representations on the held out test set
      titles=["Supervised network encoder RSM",
              "Untrained network encoder RSM"],  # plot titles
      sorting_latent=sorting_latent,
      )

  return untrained_encoder


## Uncomment below to test your function
with plt.xkcd():
  untrained_encoder = plot_rsms(seed=SEED)