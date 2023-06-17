def vae_train_loss(num_epochs, seed):
  """
  Helper function to plot the train loss of the variational autoencoder (VAE)

  Args:
    num_epochs: Integer
      Number of the epochs the VAE is to be trained for
    seed: Integer
      The seed value for the dataset/network

  Returns:
    vae_loss_array: List
      Loss per epoch
  """
  # Call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)
  # Train an encoder and classifier on the images, using models.train_classifier()
  print("Training a classifier on the pre-trained VAE encoder representations...")
  _, vae_loss_array, _, _ = models.train_classifier(
      encoder=vae_encoder,
      dataset=dSprites_torchdataset,
      train_sampler=train_sampler,
      test_sampler=test_sampler,
      freeze_features=True, # Keep the encoder frozen while training the classifier
      num_epochs=num_epochs,
      verbose=True # Print results
      )

  # Plot the VAE classifier training loss.
  fig, ax = plt.subplots()
  ax.plot(vae_loss_array)
  ax.set_title("Loss of classifier trained on a VAE encoder")
  ax.set_xlabel("Epoch number")
  ax.set_ylabel("Training loss")

  return vae_loss_array



# Set a reasonable number of training epochs
num_epochs = 25
## Uncomment below to test your function
with plt.xkcd():
  vae_loss_array = vae_train_loss(num_epochs=num_epochs, seed=SEED)