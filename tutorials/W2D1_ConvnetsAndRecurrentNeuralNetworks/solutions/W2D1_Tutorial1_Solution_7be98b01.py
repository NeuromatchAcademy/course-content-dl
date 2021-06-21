
"""
It will detect vertical edges.

Consider a part of the image where values on the left are high and values on the
right are low. In this case, applying the kernel, the high values will be
maintained and the low values will be inverted, yielding a very-positive sum.

Now reverse the situation - values on the left are low and values on the right
are high. In this case, applying the kernel, the low values will be maintained
and the high values will be inverted, yielding a very-negative sum.

In any situation on the image where the left and right sides of a 2x2 block
/aren't/ much different, applying the kernel will result in decent degree of
cancellation - i.e. if it's all, say, 5, (5 + -5 + 5 + -5) = 0.

In other words, the only situations where you'll have an extremely positive or
extremely negative score are when you have a high value on either the left/right
and a low value on the other side.
"""