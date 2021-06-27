def plot_simclr(num_epochs, seed):
  # call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)
  # EXERCISE: Train an encoder and classifier on the images, using models.train_classifier()
  print("Training a classifier on the pre-trained SimCLR encoder representations...")
  _, simclr_loss_array, _, _ = models.train_classifier(
      encoder=simclr_encoder,
      dataset=dSprites_torchdataset,
      train_sampler=train_sampler,
      test_sampler=test_sampler,
      freeze_features=True, # keep the encoder frozen while training the classifier
      num_epochs=num_epochs,
      verbose=True
      )
  # EXERCISE: Plot the loss array
  fig, ax = plt.subplots()
  ax.plot(simclr_loss_array)
  ax.set_title("Loss of classifier trained on a SimCLR encoder.")
  ax.set_xlabel("Epoch number")
  ax.set_ylabel("Training loss")


# Set a reasonable number of training epochs
num_epochs = 25
## Uncomment below to test your function
with plt.xkcd():
  plot_simclr(num_epochs=num_epochs, seed=SEED)