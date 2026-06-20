
"""
These filters are tailored to have strong responses to the features of an X and
weaker responses to the features of an O. The diagonal edge filters emphasize
the diagonal edges of the X and the checkerboard pattern reacts strongly to the
center of the X, while the more round edges of the O evoke weaker responses.

So the images which have strong downward (filter 1) and upward diagonals
(filter 3), as well as an intersection of both (filter 2) are probably an X.
""";