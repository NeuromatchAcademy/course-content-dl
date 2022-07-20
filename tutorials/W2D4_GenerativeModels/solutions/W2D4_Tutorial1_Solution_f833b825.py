
"""
With low truncation values, the images look more realistic but
all 4 images look more similar to each other. With high truncation values, the images look
less realistic but more variable. The truncation value basically  determines our desired
trade-off between the quality of generated images and the variety.

There is no easy answer to this trade-off but it's a big one in image generation! We
don't want all our generated images to look identical but to have more variety, we're
going to need to allow the image quality to drop a little.
""";