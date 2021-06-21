
"""
If the kernel were transposed, it would detect horizontal edges.

Running it on the vertical edge would result in 0s - you'd have no change
going from top to bottom, so you'd be adding and subtracting the same value.

(If you're confused as to why this shows up as black, rather than gray,
it's because of how the image is normalized)
"""