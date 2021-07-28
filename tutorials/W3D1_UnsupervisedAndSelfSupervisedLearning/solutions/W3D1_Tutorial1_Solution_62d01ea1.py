
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
""";