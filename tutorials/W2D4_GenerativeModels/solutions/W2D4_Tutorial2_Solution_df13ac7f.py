"""
Discussion:

Direction of the score points towards the higher density part of the distribution (up-hill of the landscape)
Magnitude of the score tell us how far are we from the mode, the farther the larger.

For multi-modal distribution e.g. GMM, score is the weighted average of score of
individual modes the weights is proportional to the density of each mode at this point
So the score of each point is dominated by the 'closest' modes / data points.
""";