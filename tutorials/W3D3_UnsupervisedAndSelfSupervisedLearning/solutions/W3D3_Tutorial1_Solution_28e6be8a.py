
"""
A. The classifier trained with the random encoder appears to perform a bit
   better than the classifier trained directly on the raw data.

B. The classifier trained with the random encoder performs substantially worse
   than the classifier trained along with the encoder (supervised encoder).

C. The random encoder projects the raw data to a lower dimensional
   feature space (84 features instead of 64 x 64 pixels).
   This **reduction in dimensionality**, as well as the possibility that some
   of the features **may randomly carry some shape-relevant information**,
   may explain the slight improvement in classification performance over training
   directly on the raw data. However, since the features are random,
   they are far less useful for the shape classification task than the
   supervised encoder's features which were specifically tuned to that task.
""";