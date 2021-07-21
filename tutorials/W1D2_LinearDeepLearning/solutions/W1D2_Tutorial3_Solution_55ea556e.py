def linear_regression(X, Y):
  """Analytical Linear regression

  Args:
    X (np.ndarray): design matrix
    Y (np.ndarray): target ouputs

  return:
    np.ndarray: estimated weights (mapping)
  """
  assert isinstance(X, np.ndarray)
  assert isinstance(Y, np.ndarray)
  M, Dx = X.shape
  N, Dy = Y.shape
  assert Dx == Dy

  W = Y @ X.T @ np.linalg.inv(X @ X.T)

  return W


W_true = np.random.randint(low=0, high=10, size=(3, 3)).astype(float)

X_train = np.random.rand(3, 37)  # 37 samples
noise = np.random.normal(scale=0.01, size=(3, 37))
Y_train = W_true @ X_train + noise

## Uncomment and run
W_estimate = linear_regression(X_train, Y_train)
print(f"True weights:\n {W_true}")
print(f"\nEstimated weights:\n {np.round(W_estimate, 1)}")