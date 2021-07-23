
"""
1
  - Very large mini-batch sizes (>= 2048) -- along with synchronous batch
    normalization across all GPUs / mini-batches to address the high diversity
    of the ImageNet dataset
  - Increased number of parameters (at least double of previous state-of-the-art
    models like SAGAN) (from increased widths of the layers)
  - Spectral normalization, orthogonal regularization penalty, and dropouts
    in final layer of D to stabilize the highly unstable training of BigGAN.
    Even with these stabilization techniques, BigGAN still suffers from chronic
    instability (as described in the paper) and requires multiple restarts.
    The model almost always collapses at some point in the training, but the key
     is to stop the training before it does.

2. Huge compute requirement to allow for large mini-batch sizes and therefore
   high energy consumption

3. N/A
"""