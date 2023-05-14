
"""
1. Cosine similarity is a metric used to measure how similar the documents
   are, irrespective of their size. Mathematically, it calculates the cosine of the angle
   between two vectors projected in a multi-dimensional space, i.e., when plotted
   on a multi-dimensional space. Word embedding is a word representation in
   a vector space. Thus, the cosine similarity captures the orientation (the angle) of
   the words, irrespective of their magnitude. Thus, high cosine similarity
   implies that the words in the latent space share a similar context.

2. The words in the key closer to the centroid mean that the words strongly
   co-occur in the same context as the key,
   i.e., the correlation/representativeness between the keys closely clustered
   with the centroid is higher than the correlation in the more diversified clusters.
""";