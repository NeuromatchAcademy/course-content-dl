def train_supervised_encoder(num_epochs, seed):
  """
  Helper function to train the encoder in a supervised way

  Args:
    num_epochs: Integer
      Number of epochs the supervised encoder is to be trained for
    seed: Integer
      The seed value for the dataset/network

  Returns:
    supervised_encoder: nn.module
      The trained encoder with mentioned parameters/hyperparameters
  """
  # Call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)

  # Initialize a core encoder network on which the classifier will be added
  supervised_encoder = models.EncoderCore()
  # Train an encoder and classifier on the images, using models.train_classifier()
  print("Training a supervised encoder and classifier...")
  _ = models.train_classifier(
      encoder=supervised_encoder,
      dataset=dSprites_torchdataset,
      train_sampler=train_sampler,
      test_sampler=test_sampler,
      freeze_features=False,
      num_epochs=num_epochs,
      verbose=True  # print results
      )

  return supervised_encoder


# Add event to airtable
atform.add_event('Coding Exercise 1.2.1: Training a logistic regression classifier along with an encoder')

num_epochs = 10  # Proposed number of training epochs
## Uncomment below to test your function
supervised_encoder = train_supervised_encoder(num_epochs=num_epochs, seed=SEED)