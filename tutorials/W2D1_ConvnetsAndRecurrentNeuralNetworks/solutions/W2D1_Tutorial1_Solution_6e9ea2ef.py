
"""
1. The overfitting is now reduced, as the validation loss now does not increase
   strongly anymore. But as the training loss still decreases while the
   validation loss stays constant, the network is still overfitting, but this
   time it at least does not affect the generalisation performance as strongly
   as before.

2. Since dropout is only applied to training process, the training accuracy is
   reduced but not the validation accuracy.
""";