
"""
The problem here is that some words might be just more frequent than the others. The authors
of the CrowS-Pairs paper (https://github.com/nyu-mll/crows-pairs) create a more sophisticated metric
as shown: score = \sum _{i=0} ^ {|C|} log P (u_i ∈ U|U_{u_i}, M, θ)
where U is the set of unmasked tokens, M is the set of masked tokens.
For each sentence, we mask one unmodified token at a time until all unmasked tokens have been masked.
However, in this section for simplicity
we computed raw probabilities. That is okay since we
intentionally chose the words that have roughly the same distribution.
""";