
"""
1. The ResNeXt has 32 * 4 = 128 channels in the bottleneck, whereas the ResNet has only 64.

2. In ResNet all 64 output channels of the first layer are connected to all 64 output channels of the second layer.
In ResNeXt, channels are connected only within paths, i.e. in groups of 4.

3. ResNeXt contains more channels in the bottleneck (potentially more expressive),
   but each of them "sees" only a subset of the previous layer's output (potentially less expressive).
   The former tends to outweigh the latter, which is why the ResNeXt architecture tends to outperform the
   vanilla ResNet architecture.
""";