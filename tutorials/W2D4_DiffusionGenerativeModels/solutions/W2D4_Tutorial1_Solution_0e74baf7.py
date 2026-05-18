
"""
The images generated from the VAE look far more like real handwritten digits.

An Autoencoder accepts input, compresses it, and recreates it. On the other hand,
VAEs assume that the source data has some underlying distribution and attempts
to find the distribution parameters. Basically, VAEs are built to allow image generation
and AEs are not, so it's not surprising that VAEs generate more realistic images!

VAEs are similar to GANs, although as we will see in the next tutorials, GANs work
a bit differently.
""";