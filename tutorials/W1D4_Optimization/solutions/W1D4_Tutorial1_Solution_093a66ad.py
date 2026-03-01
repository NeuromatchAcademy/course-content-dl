
"""
  Remember the discussion in Section 1 about surrogate objectives.
Our optimization methods minimize the loss, but at the end of the day we care about test accuracy.

  However, we can't directly optimize for test accuracy and the finite size of our
datasets lead us to (cross-)validation:

  1. We minimize the loss (empirical risk minimization) on our *training set*.
  2. We choose models and hyperparameters on the *validation set*.
  3. We use the *test set* in order to report the final performance of our model on unseen data.
""";