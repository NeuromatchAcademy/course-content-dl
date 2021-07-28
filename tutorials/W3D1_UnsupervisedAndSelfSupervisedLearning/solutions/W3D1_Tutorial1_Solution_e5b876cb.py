
"""
A. The classifiers trained on top of the random and pre-trained VAE encoders
   perform poorly (below 50%) on the shape classification task, regardless of
   how much of the labelled data is available to train the classifier.
   In contrast, the classifier trained on top of the pre-trained
   **SimCLR encoder maintains a performance above 90%** even when it is
   trained with only 5% of the total available labelled data. The classifier
   trained along with the **supervised encoder is most heavily affected**, with
   its performance dropping to about 45% when it is trained with only 5% of the
   total available data.

B. Since the **supervised encoder** cannot be trained with unlabelled data,
   any reduction in the fraction of labelled data available effectively means a
   reduction in number of training examples. If the number of training examples
   available is quite small, the encoder may not be able to learn generalizable
   features. In contrast, since the task used to train the **SimCLR encoder does
   not require any labels**, the encoder can first be pre-trained on the full dataset
   to learn generalizable features that are relevant to the downstream classification,
   as was done in the previous sections. If this is successful, the classifier layer
   training becomes relatively simple. Indeed, if the encoder already broadly decodes
   shapes, the classifier layer needs only to learn to map shape-like feature
   to the correct shape. As the results here show, such a simple task likely
   requires far fewer training examples than training a full encoder network
   and classifier from scratch. Importantly, this is not a trivial finding,
   as shown by the overall poor shape classification performance of the
   classifiers trained on the **random** and **pre-trained VAE** encoders.
   The fact that increasing the number of labelled examples available barely
   impacts their performance suggests that their features, in contrast to the
   SimCLR encoder's features, are not very informative at all about shape.
""";