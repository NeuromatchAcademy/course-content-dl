def plot_loss(num_epochs, seed):
  # call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)

  # train classifier on the randomly encoded images
  print("Training a classifier on the untrained encoder representations...")
  _, untrained_loss_array, _, _ = models.train_classifier(
      encoder=untrained_encoder,
      dataset=dSprites_torchdataset,
      train_sampler=train_sampler,
      test_sampler=test_sampler,
      freeze_features=True,  # keep the encoder frozen while training the classifier
      num_epochs=num_epochs,
      verbose=True  # print results
      )
  # EXERCISE: Plot the loss array
  fig, ax = plt.subplots()
  ax.plot(untrained_loss_array)
  ax.set_title("Loss of classifier trained on an untrained encoder.")
  ax.set_xlabel("Epoch number")
  ax.set_ylabel("Training loss")

  return untrained_loss_array


# EXERCISE: Set a reasonable number of training epochs
num_epochs = 25
## Uncomment below to test your plot
with plt.xkcd():
  untrained_loss_array = plot_loss(num_epochs=num_epochs, seed=SEED)