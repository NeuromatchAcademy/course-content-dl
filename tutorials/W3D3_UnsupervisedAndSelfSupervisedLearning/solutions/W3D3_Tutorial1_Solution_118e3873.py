
"""
A.  A few examples of principles we can draw on to successfully apply
    SSL to learning good data representations include:
    - When limited labelled data is available, model pre-training
      can greatly enhance the quality of the feature space a model learns.
    - Data augmentations selection can be guided, at least in part,
      by an understanding of both:
      - the types of latent variables that underlie the data of interest
      - the types of latent variables our encoder might need to learn as
        features (e.g., shape, orientation) or become invariant to
        (e.g., color, motion, scale) in order to perform well on specific
        types of downstream tasks.

B. A few examples of challenges that we might face include:
    - Identifying useful data augmentations to learn good features for a more
      complex dataset (e.g., a biosignal dataset, a speech dataset).
    - Anticipating the types features that might be relevant to more complex
      downstream tasks (e.g., detecting pathological biosignals, classifying
      speaker identity).
    - Allocating additional training time and resources, as SSL tasks are
      typically more computationally demanding than their supervised
      counterparts.

C. When working with a non visual dataset, e.g., a speech dataset, different
   types of augmentations are needed from the ones used in this tutorial. The
   type of augmentations used will still depend on the downstream task of
   interest, of course. If the downstream task of interest is something like
   speech to text translation, it could be useful to learn representations
   that are invariant to the pitch of a voice. So, one could design an
   augmentation that randomly shifts the pitch of a speech sample. If the
   downstream task is pitch sensitive, however, like speaker identification,
   a different augmentation could be designed, like a time shift where two
   samples close in time are positive pairs for each other.

D. For sequential or time series data, one could use a very different type of
   SSL task, like a predictive task. In such a task, a network could be trained
   to predict the representation of time point t_2 from the representation of
   time point t_1. In order to successfully accomplish this task of predicting
   electrical brain activity representations sequentially, our network would
   likely learn data representations that change gradually and predictably
   in time. Since the stages of sleep also change gradually through time, this
   network would have a good chance of being successful in downstream tasks
   that rely on temporal features, like sleep staging. This type of
   predictive SSL is applied in a more sophisticated way in algorithms like
   Contrastive Predictive Coding (CPC), for example.
""";