def plot_rsms2():
  sorting_latent = "shape"
  # EXERCISE: Visualize RSMs for the supervised, VAE and SimCLR network encoders.
  print("Plotting RSMs...")
  _ = models.plot_model_RSMs(
      encoders=[supervised_encoder, vae_encoder, simclr_encoder],
      dataset=dSprites_torchdataset,
      sampler=test_sampler, # we want to see the representations on the held out test set
      titles=["Supervised network encoder RSM", "VAE network encoder RSM",
              "SimCLR network encoder RSM"], # plot titles
      sorting_latent=sorting_latent
      )


## Uncomment below to test your code
with plt.xkcd():
  plot_rsms2()