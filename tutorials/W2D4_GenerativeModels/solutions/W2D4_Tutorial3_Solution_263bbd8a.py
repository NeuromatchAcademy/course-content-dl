
"""
In Stable Diffusion, the (time_proj) and (time_embedding) layers resemble the GaussianFourierProjection layers we used in the last coding exercise, both multiplex time t and fed them into a neural network to modulate the convolutional layers per channel.
(down_blocks) resembles the down sampling convolution layers from conv1 to conv4 .
(up_blocks) resembles the up sampling and transpose convolution layers we from tconv4 to tconv1 .
text_model is a CLIP model that encodes text conditional signal into a bunch of word vectors.
CrossAttention in Stable Diffusion implements conditional modulation.
""";