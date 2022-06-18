
"""
A. As seen in the much yellower RSMs of the SimCLR encoder trained with only
   2 negative pairs used in the contrastive loss, reducing the number of
   negative pairs leads to a **substantial increase in the density of high
   feature similarity values** (near 1). This is quantified in the histogram,
   which shows that the probability density of similarity values above 0.5
   increases considerably, with the density of values of almost 1 increasing
   3-4x (from 1 to 3.5). If we look at the shape RSM, we see that this results
   in a loss of the **distinction between squares and ovals** in feature space.
   Interestingly, given that a few negative pairs are still included in the
   contrastive loss, the observed increase in high similarity values is
   counterbalanced by a concurrent increase in strongly negative similarity
   values (near -1).

B. The shape classifier is likely to still classify hearts reasonably well,
   but to do a poor job of distinguishing ovals from squares.

C. Negative pairs are used, in contrastive models like SimCLR, as a
   **counterweight to positive pairs**. Indeed, if only positive pairs
   were used to train a contrastive model, the network could settle on a
   trivial solution: a **collapsed feature space where all or most images
   are encoded identically**. Such a feature space would be entirely useless,
   as it would not preserve any information about the input data. To prevent
   this, it is important to ensure that while the network updates its weights
   to encode positive pairs similarly, **it still generally encodes other,
   randomly selected pairs of images distinctly**. The negative pairs are
   therefore used to obtain an estimate of how distinctly the network would
   encode random pairs of images. If this sample is too small
   (as with our SimCLR encoder trained with a loss calculated
   from only 2 negative pairs),
   the estimate will likely not be representative at all. In this case,
   although the network may still encode a few pairs as highly dissimilar,
   it still runs the risk of learning a partially collapsed feature space,
   as observed above. Our normal SimCLR encoder is trained with a loss calculated
   from far more (999) negative pairs, and this enables it to learn a much more
   distributed and meaningful feature space.
""";