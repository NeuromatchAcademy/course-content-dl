
"""
As you have noticed, all of the attention values are quite similar, even for very dissimilar words. There are two reasons for this.

1. for simplicity we used a very small corpus. As the corpus size increases, we can use a larger embedding size. With a larger embedding,
the similarity between any two words decreases ON AVERAGE, which will increase the specificity of self-attention

2. we only used one round of self attention. As you will see in the upcoming transformer architecture,
we often use multiple transformer blocks sequentially.

""";