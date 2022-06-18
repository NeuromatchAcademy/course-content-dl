
"""
Dropout is an effective regularization technique but may hurt model performance
if the model does not overfit i.e., low model capacity.

Placement of dropout within the model matters.
For instance,
Dropout is generally not applied immediately before the last layer,
because the network has no ability to "correct" errors induced by
dropout before the classification happens.
Additionally, if the model is not trained until convergence,
dropout may give worse results. Usually dropout hurts performance at the start
of training, but results in the final "converged" error being lower.
""";