def shuffle_and_split_data(X, y, seed):
  # set seed for reproducibility
  torch.manual_seed(seed)
  # Number of samples
  N = X.shape[0]
  # Shuffle data
  shuffled_indices = torch.randperm(N)   # get indices to shuffle data, could use torch.randperm
  X = X[shuffled_indices]
  y = y[shuffled_indices]

  # Split data into train/test
  test_size = int(0.2 * N)    # assign test datset size using 20% of samples
  X_test = X[:test_size]
  y_test = y[:test_size]
  X_train = X[test_size:]
  y_train = y[test_size:]

  return X_test, y_test, X_train, y_train


## Uncomment below to test your function
X_test, y_test, X_train, y_train = shuffle_and_split_data(X, y, seed=SEED)
with plt.xkcd():
  plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
  plt.title('Test data')
  plt.show()