def plot_rsms_all():
  sorting_latent = "shape"
  print("Plotting RSMs...")
  # EXERCISE: Visualize RSMs for the supervised, untrained and VAE network encoders.
  _ = models.plot_model_RSMs(
      encoders=[supervised_encoder, untrained_encoder, vae_encoder], # we pass all three encoders
      dataset=dSprites_torchdataset,
      sampler=test_sampler, # we want to see the representations on the held out test set
      titles=["Supervised network encoder RSM", "Untrained network encoder RSM",
              "VAE network encoder RSM"], # plot titles
      sorting_latent=sorting_latent,
      )


## Uncomment below to test your function
with plt.xkcd():
  plot_rsms_all()