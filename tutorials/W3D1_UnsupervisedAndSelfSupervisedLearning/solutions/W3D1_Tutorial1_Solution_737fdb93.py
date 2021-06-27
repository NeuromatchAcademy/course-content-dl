def train_supervised_encoder(num_epochs, seed):
  # call this before any dataset/network initializing or training,
  # to ensure reproducibility
  set_seed(seed)

  # initialize a core encoder network on which the classifier will be added
  supervised_encoder = models.EncoderCore()
  # EXERCISE: Train an encoder and classifier on the images, using models.train_classifier()
  print("Training a supervised encoder and classifier...")
  _ = models.train_classifier(
      encoder=supervised_encoder,
      dataset=dSprites_torchdataset,
      train_sampler=train_sampler,
      test_sampler=test_sampler,
      freeze_features=False, # we train the encoder, along with the classifier layer
      num_epochs=num_epochs,
      verbose=True # print results
      )

  return supervised_encoder

num_epochs = 10 # Proposed number of training epochs
## Uncomment below to test your function
supervised_encoder = train_supervised_encoder(num_epochs=num_epochs, seed=SEED)