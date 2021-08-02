
"""
A. The yellow diagonal corresponds to the similarity between each encoded image and itself.
   Since each encoded image is, of course, identical to itself, the similarity
   is of 1 at each point on the diagonal.

B. The pattern we observe is that there are square sections of the RSM that have
   higher similarity values than the rest, and these sections lie along the
   yellow diagonal.
   These sections correspond to the similarities between **encoded images of the
   same shape** (e.g., 2 hearts), which are generally **higher than the similarities
   between encoded images of different shapes** (e.g., a heart and a square),
   when using this trained, supervised encoder.

C. It is a bit subtle, but it looks like the **hearts and squares** might be
   encoded more similarly to one another than the **ovals and squares**, in general.
   This is based on the fact that the RSM values for hearts x squares
   (bottom left and top right) appear to be lighter (more yellow) than the RSM
   values for ovals x squares (top middle and middle left),
   which are a bit darker (more blue).

D. If we sort by different latent dimensions (e.g., `scale`, `orientation`, `posX` or `posY`),
   we do not see as much structure in the RSMs. This is because the supervised
   encoder is specifically trained on a shape classification task, which forces
   it to encode images of the same shape more similarly, and images of different
   shapes more differently. It is not trained to distinguish scales, orientations
   or positions. If it **were** trained to predict `orientation`, `scale` or `position`,
   we could expect to see similar RSM patterns, with high similarity along the
   diagonal for the predicted dimension.
""";