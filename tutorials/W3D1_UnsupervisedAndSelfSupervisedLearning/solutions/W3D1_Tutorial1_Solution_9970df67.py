
"""
A. Only the yellow diagnonal identity line is visible. No other patterns emerge,
   as most images are encoded with near 0 similarity to one another, using the random encoder.

B. The trained, supervised network produces more meaningful representations,
   as the similarity between different encoded images actually
   **captures certain meaningful conceptual similarities between the different images**,
   specifically shape similarities. In other words, the image representations
   obtained with the trained, supervised encoding reflect the fact that two hearts
   are more conceptually similar to each other in terms of shape than a heart and a square.
"""