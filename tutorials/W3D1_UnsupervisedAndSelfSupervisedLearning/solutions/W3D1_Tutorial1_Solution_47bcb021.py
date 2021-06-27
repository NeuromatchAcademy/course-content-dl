def sorting_plot():
  sorting_latent = "shape" # Exercise: Try sorting by different latent dimensions
  # EXERCISE: Visualize RSMs for the normal SimCLR, 2-neg-pair SimCLR and untrained network encoders.
  print("Plotting RSMs...")
  simclr_rsm, simclr_neg_pairs_rsm, untrained_rsm = models.plot_model_RSMs(
      encoders=[simclr_encoder, simclr_encoder_neg_pairs, untrained_encoder],
      dataset=dSprites_torchdataset,
      sampler=test_sampler, # we want to see the representations on the held out test set
      titles=["SimCLR network encoder RSM",
              f"SimCLR network encoder RSM\n(2 negative pairs per image used in loss calc.)",
              "Untrained network encoder RSM"], # plot titles
      sorting_latent=sorting_latent
      )
  # EXERCISE: Plot a histogram of RSM values for both encoders.
  plot_rsm_histogram(
      [simclr_rsm, simclr_neg_pairs_rsm],
      colors=["gray", "royalblue"],
      labels=["normal SimCLR RSM", "few neg. pairs SimCLR RSM"],
      nbins=100
      )


## Uncommnet below to test your code
with plt.xkcd():
  sorting_plot()