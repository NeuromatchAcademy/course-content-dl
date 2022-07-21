
"""
Let's say we trained the discriminator first to distinguish between real images
and fake images perfectly. Then we train our generator model to try to fool the
discriminator.

This wouldn't work very well! The generator model would NEVER fool the discriminator if it's
already really good. If it never fools the discriminator, there won't be good information on
how to improve to fool the discriminator. Imagine you're a beginner soccer player going head-to-head
with a pro. You have no chance of making a shot so you're not really learning what works (because
nothing does).
""";