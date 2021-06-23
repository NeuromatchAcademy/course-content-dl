class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, b, h, t, k):
        """
        Compute dot products. This is the same operation for each head,
        so we can fold the heads into the batch dimension and use torch.bmm
        Note: .contiguous() doesn't change the actual shape of the data,
        but it rearranges the tensor in memory, which will help speed up the computation
        for this batch matrix multiplication.

        Shape of `queries`: (`batch_size`, no. of queries, `d`)
        Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
        Shape of `values`: (`batch_size`, no. of key-value pairs, value dimension)

        b: batch size
        h: number of heads
        t: number of keys/queries/values (for simplicity, let's assume they have the same sizes)
        k: embedding size
        """
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # Matrix Multiplication between the keys and queries
        score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(k) # size: (b * h, t, t)
        softmax_weights = F.softmax(score, dim=2) # row-wise normalization of weights

        # Matrix Multiplication between the output of the key and queries multiplication and values.
        out = torch.bmm(self.dropout(softmax_weights), values).view(b, h, t, k) # rearrange h and t dims
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return out