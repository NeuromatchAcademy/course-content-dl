
"""
A. The reconstructions preserve most of the latent features well:
  - shape: ovals shapes are well preserved, whereas squares lose some sharpness
    in their edges. Hearts are poorly preserved, as the sharp angles of their edges are lost.
  - scale: shape scales are well preserved.
  - orientation: orientations are well preserved.
  - posX and posY: shape positions are very well preserved.

B. Since several of the latent features of the images are well preserved
   in the reconstructions, it is possible that the VAE encoder has indeed
   learned a feature space very similar to the known latent dimensions of the data.
   However, it is also possible that the VAE encoder instead learned a different
   latent feature space that is good enough to achieve reasonable image reconstruction.
   Examining the RSMs should shed light on that.
""";