
"""
1. 2D as it is simply a more difficult distribution to model compared to the
  unimodal 1D example.
2. The training becomes unstable and eventually diverges. Yes.
3. Training is too slow and takes longer to convege.

In general, when training a GAN, we need to ensure a certain balance between
the critic and generator training. As such, tuninig the learning rate
is crucial for succesfully training GANs.
""";