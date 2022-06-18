def plot_loss(num_epochs, seed):
  """
  Helper function to plot the loss function of the random-encoder

  Args:
    num_epochs: Integer
      Number of the epochs the random encoder is to be trained for
    seed: Integer
      The seed value for the dataset/network

  Returns:
    random_loss_array: List
      Loss per epoch
  """
  # Call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)

  # Train classifier on the randomly encoded images
  print("Training a classifier on the random encoder representations...")
  _, random_loss_array, _, _ = models.train_classifier(
      encoder=random_encoder,
      dataset=dSprites_torchdataset,
      train_sampler=train_sampler,
      test_sampler=test_sampler,
      freeze_features=True,  # Keep the encoder frozen while training the classifier
      num_epochs=num_epochs,
      verbose=True  # Print results
      )
  # EXERCISE: Plot the loss array
  fig, ax = plt.subplots()
  ax.plot(random_loss_array)
  ax.set_title("Loss of classifier trained on a random encoder.")
  ax.set_xlabel("Epoch number")
  ax.set_ylabel("Training loss")

  return random_loss_array


# Add event to airtable
atform.add_event('Coding Exercise 3.1.2: Evaluating the classification performance of a logistic regression')

## Set a reasonable number of training epochs
num_epochs = 25
## Uncomment below to test your plot
with plt.xkcd():
  random_loss_array = plot_loss(num_epochs=num_epochs, seed=SEED)