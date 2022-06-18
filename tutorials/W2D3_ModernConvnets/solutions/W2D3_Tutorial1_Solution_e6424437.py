
"""
1. The closer your pretraining and target data domains are, the better pretraining will work
2. The more pretraining data you have, the better pretraining will work
3. The better your model is able to take advantage of your pretraining data (that is to say
   the larger your model is as you have enough data), the better pretraining will work

Pretraining isn't necessarily always a benefit though. If your source domain is very different from
the domain you're trying to predict, your models might learn unhelpful features.

Additionally, if you have a lot of training data in your target domain, pretraining data might
cause your model to converge to a local minimum (this process is referred to as ossification in
the Scaling Laws for Transfer paper cited in the Further Reading section)
""";