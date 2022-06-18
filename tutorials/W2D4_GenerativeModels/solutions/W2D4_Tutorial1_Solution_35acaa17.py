
"""
Single layered Autoencoder with linear activation function is similar to PCA.
However, the Autoencoder can better capture the curvature in the data and hence,
attempts to encode the most important features since they are capable of
reconstruction anyway but are also prone to overfitting due to a higher number of parameters.
PCA retains projections onto planes with maximum variance (and minimum error) and loses data.
The Autoencoder performs poorly if there are no underlying relationships between the features.
""";