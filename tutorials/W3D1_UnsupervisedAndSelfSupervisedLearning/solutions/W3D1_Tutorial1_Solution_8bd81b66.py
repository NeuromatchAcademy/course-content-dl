def plot_rsms(seed):
  # call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)

  # initialize a core encoder network that will not get trained
  random_encoder = models.EncoderCore()

  # EXERCISE: Try sorting by different latent dimensions
  sorting_latent = "shape"

  # EXERCISE: Plot RSMs
  print("Plotting RSMs...")
  _ = models.plot_model_RSMs(
      encoders=[supervised_encoder, random_encoder],  # we pass both encoders
      dataset=dSprites_torchdataset,
      sampler=test_sampler,  # we want to see the representations on the held out test set
      titles=["Supervised network encoder RSM",
              "Random network encoder RSM"],  # plot titles
      sorting_latent=sorting_latent,
      )

  return random_encoder


# add event to airtable
atform.add_event('Coding Exercise 3.1.1: Plotting a random network encoder RSM along different latent dimensions')

## Uncomment below to test your function
with plt.xkcd():
  random_encoder = plot_rsms(seed=SEED)