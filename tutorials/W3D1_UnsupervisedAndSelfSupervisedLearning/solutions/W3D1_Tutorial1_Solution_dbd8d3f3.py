def train_plot_simclr(num_epochs, seed):
  # call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)
  print("Training a classifier on the representations learned by the SimCLR "
        "network encoder pre-trained\nusing only 2 negative pairs per image "
        "for the loss calculation...")
  # EXERCISE: Train an encoder and classifier on the images, using models.train_classifier()
  _, simclr_neg_pairs_loss_array, _, _ = models.train_classifier(
    encoder=simclr_encoder_neg_pairs,
    dataset=dSprites_torchdataset,
    train_sampler=train_sampler,
    test_sampler=test_sampler,
    freeze_features=True, # keep the encoder frozen while training the classifier
    num_epochs=num_epochs,
    verbose=True
    )

  # Plot the loss array
  fig, ax = plt.subplots()
  ax.plot(simclr_neg_pairs_loss_array)
  ax.set_title(("Loss of classifier trained on a SimCLR encoder\n"
  "trained with 2 negative pairs only."))
  ax.set_xlabel("Epoch number")
  _ = ax.set_ylabel("Training loss")

  return simclr_neg_pairs_loss_array


# Set a reasonable number of training epochs
num_epochs = 50
## Uncomment below to test your code
with plt.xkcd():
  _ = train_plot_simclr(num_epochs=num_epochs, seed=SEED)