
"""
Note how gradient descent (with a properly tuned step size) guarantees an improvement
in the objective function, unlike the random udpate method, which very rarely finds a
direction that improves the objective function. The green line indicates
a change of zero in the value of the loss: we want to be below zero to minimize!

The magnitue of the improvement at initialization is much bigger than
the change in loss obtained at a later stage in training. This is expected,
as we have gotten closer to the optimum parameter configuration.
""";