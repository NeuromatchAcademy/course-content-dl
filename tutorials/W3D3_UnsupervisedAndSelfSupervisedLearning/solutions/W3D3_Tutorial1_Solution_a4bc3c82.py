def rsms_and_histogram_plot():
  """
  Function to plot Representational Similarity Matrices (RSMs) and Histograms

  Args:
    None

  Returns:
    Nothing
  """
  sorting_latent = "shape" # Exercise: Try sorting by different latent dimensions
  # EXERCISE: Visualize RSMs for the normal SimCLR, 2-neg-pair SimCLR and random network encoders.
  print("Plotting RSMs...")
  simclr_rsm, simclr_neg_pairs_rsm, random_rsm = models.plot_model_RSMs(
      encoders=[simclr_encoder, simclr_encoder_neg_pairs, random_encoder],
      dataset=dSprites_torchdataset,
      sampler=test_sampler, # To see the representations on the held out test set
      titles=["SimCLR network encoder RSM",
              f"SimCLR network encoder RSM\n(2 negative pairs per image used in loss calc.)",
              "Random network encoder RSM"], # Plot titles
      sorting_latent=sorting_latent
      )
  # EXERCISE: Plot a histogram of RSM values for both SimCLR encoders.
  plot_rsm_histogram(
      [simclr_neg_pairs_rsm, simclr_rsm],
      colors=["royalblue", "gray"],
      labels=["few neg. pairs SimCLR RSM", "normal SimCLR RSM"],
      nbins=100
      )

## Uncomment below to test your code
with plt.xkcd():
  rsms_and_histogram_plot()