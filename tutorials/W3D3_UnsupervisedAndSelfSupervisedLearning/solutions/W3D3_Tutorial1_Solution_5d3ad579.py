
"""
A. The RSMs for the pre-trained SimCLR encoder show that it
   **encodes `shape` almost as strongly** as the supervised encoder.
   Unlike the supervised encoder, it also appears to encode `scale` quite strongly.
   In addition, unlike the pre-trained VAE encoder, it does not appear to
   encode position strongly.

B. The RSM structures observed for the pre-trained SimCLR encoder strongly
   reflect the transformations selected as augmentations during training.
   Indeed, these augmentations are closely related to several of the original
   latent dimensions used to sort the RSMs:
    - `scale` for `scale`,
    - `degrees` for `orientation`, and
    - `translation` for `posX` and `posY`

   This likely explains why almost no structure is visible in the RSMs sorted by
   `orientation`, `posX` and `posY`: **the encoder was specifically trained to
   ignore these features**, i.e. to learn a feature space that is invariant to
   differences along these dimensions. As a result of this training,
   the only original latent dimension left for the encoder to encode in
   feature space was `shape`.
   Interestingly, although the SimCLR encoder was also trained to ignore `scale`,
   it converged on a solution to the contrastive task that still encodes
   `scale` to some extent.

C. The supervised encoder is likely the best encoder for tasks that rely on
   distinguishing shapes similar to those in the dataset. The SimCLR encoder is
   likely useful for tasks that rely on differentiating shape and/or scale.
   Lastly, the VAE encoder is likely the most useful of the three for position
   decoding, as well as image reconstruction, when paired with the VAE decoder
   it was trained with.

D. Good performance by a SimCLR encoder on a contrastive task does **not** guarantee
   that the encoder will perform well on a downstream classification task.
   The performance of the encoder on the contrastive task will likely only reflect
    future classification performance if the contrastive task has been designed
    in a way that **specifically promotes learning of a feature space that is
    relevant to that downstream classification task**. For example, here the
    contrastive task drove the encoder towards becoming invariant to all latent dimensions
    other than shape, which is intuitively exactly what is needed for a shape
    classification task.
    If other, less relevant augmentations had been selected instead
    (e.g., adding random noise or inverting the colors), an encoder with the
    same amount of pre-training might have performed very poorly on the
    downstream shape classification task.

E. If we wanted to pre-train a SimCLR encoder to decode `orientation` instead of `shape`,
   we would likely **remove the `degrees` augmentation**, as it drives the encoder
   to become invariant to the different orientations of the shapes.
   To support the network's ability to generalize to new shapes, we might want to
   push the encoder to be more invariant to the `shape` dimension. To do this, we
   might use some sort of **filter augmentation**, like a Gaussian, that slightly
   distorts shape edges. However, such a shape distortion augmentation would probably
   have to be applied **carefully** in order to avoid transforming all shapes into amorphous
   blobs with no discernible orientation or making shapes appear like
   **different shapes with totally different orientations.**
   Indeed, since **orientation is at least partially determined from shape**, a
   feature space that is good for predicting orientation will likely not be
   entirely invariant to shape.
""";