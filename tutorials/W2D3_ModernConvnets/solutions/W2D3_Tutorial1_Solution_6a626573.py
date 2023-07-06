
"""
1. if number of paths=1 and channels per path=64 -> same architecture as ResNet, no parameter saving
2. if number of paths=8 and channels per path=8 -> around half the number of parameters
3. if number of paths=32 and channels per path=2 -> biggest parameter saving the more paths, the more saving
""";