
"""
A. The classifier trained with the supervised encoder is most affected by the
    biased training dataset, which drops to chance performance.
    In contrast, the classifier trained with the pre-trained SimCLR encoder
    is least affected. Classifiers trained with the random or pre-trained VAE
    encoders also drop to chance performance.

B. It is likely that the drop in performance observed with the supervised
   encoder is due to that fact that test set image distribution is poorly
   represented by the training set image distribution. Indeed, certain
   `shape`/`posX` combinations which exist in the test set do not appear at all
   in the training set. As a result, it is likely that the classifier performs
   very poorly when classifying squares on the right or hearts on the left,
   as in the training set, all squares are on the left, and all hearts are
   on the right. In fact, it is possible that the network picked up on this
   biased relationship between `shape` and `posX` during training, and learned
   to classify shapes almost exclusively from their position in X in the image.
   Such a solution would generalize very poorly to the test set where this
   biased relationship does not exist.

C. The pre-trained SimCLR encoder is far more robust to this bias,
   as its pre-training is not limited to the training set images. Indeed, since
   it is trained on training set image augmentations, this forces the encoder
   to learn a feature space that captures a much broader distribution of images
   than exists in the training set alone. In this case, the data augmentations
   directly ensure that the SimCLR encoder is robust to the specific type of
   bias used, as they push the network to learn **representations that are
   invariant to position in x** (and y). As a result, when the classifier is
   trained on top of the pre-trained encoder with only the biased training set,
   it is more likely to learn the appropriate mapping from the feature space to
   shape, without interference from the correlated position information.

D. From these examples, we can see how self-supervised learning, and
   specifically data augmentation, has the potential to help mitigate the
   negative effects of biases that exist in our training sets. However, it is
   important to note that in this example, the dataset and classification task
   are very simple, and we actually know exactly what the bias in the
   training dataset is. This makes selecting appropriate data augmentations
   quite simple. In real-world scenarios, it is not so obvious.
   Some strategies for selecting good data augmentations in real-world
   scenarios might include:
    - identifying dimensions that a model should in theory be invariant to and
      designing augmentations tailored to promote invariance to these dimensions,
    - identifying **known sources** of bias, for example based on existing
      research in psychology and sociology, and tailoring data augmentations to
      these biases,
    - designing biased training datasets to evaluate how robust models are to
      these biases following training.

Of course, in addition to mitigation strategies, it is critically important that
we reduce biases at the source by improving data collection and dataset curation
strategies.
"""