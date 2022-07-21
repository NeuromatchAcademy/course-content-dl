
"""
A linear autoencoder is a very similar set-up to PCA - remember from the video that both
will find solutions on the same linear subspace.

Lower values of K result in worse reconstruction quality using both the linear autoencoder
and PCA - this is because the images are being squished through a smaller bottleneck so more
information is lost! If K = 5, each image is represented by just 5 numbers at that bottleneck,
instead of by the number of pixels there are (784 for mnist)!

I think it's pretty impressive that the reconstructions are as good as they are...
""";