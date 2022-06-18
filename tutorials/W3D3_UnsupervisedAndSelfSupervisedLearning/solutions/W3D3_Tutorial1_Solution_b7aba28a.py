
"""
A. The VAE RSMs show very little structure along the diagonal when sorted by `shape` or `orientation`,
   even though both features were reasonably well preserved in the reconstructions.
   Some weak structure may be visible for `scale`, but the **clearest structure emerges for `posX` and `posY`**.
   This structure shows that the VAE encoder encodes shapes at nearby positions
   more similarly to each other than shapes that are farther apart.
   Together, these results suggest that although **the VAE was able to reconstruct the images,
   it did not end up learning a feature space that fits the known latent dimensions** of the dataset, except position in x and y.

B. Like the random encoder, but unlike the supervised encoder, the pre-trained
   VAE encoder **shape RSM** shows no structure. However, the pre-trained
   VAE encoder's **position RSMs** do show more structure than either
   the supervised or random encoder RSMs.

C. The differences between the supervised and pre-trained VAE encoder RSMs
   reflect the fact that the encoders are trained on very different
   tasks: classification and reconstruction, respectively. With such different
   constraints and requirements, the feature space needed to accomplish these
   tasks is likely to be quite different, as is the case here. Of course, the
   random encoder is not trained at all, so its feature space is unlikely to
   randomly be similar to either the supervised or VAE encoder's feature space.

D. The pre-trained VAE encoder is not likely to perform very well on shape classification,
   given the lack of structure in its RSM.

E. The pre-trained VAE encoder might be better suited to **predicting `posX` or `posY`**,
   as the RSMs show that its feature space does encode these features to some extent.
""";