def convolution2d(image, kernel):
    """
    Convolves the provided image (a greyscale image with each entry
                                  ranging from 0 to 255)
    with the provided kernel.

    arguments:

    image: A numpy matrix, which is (a list (column) of lists (rows) of pixels)
    kernel: A numpy matrix, which is (a list (column) of lists (rows) of pixels)
    outputs: A numpy matrix, the result of convolving the kernel over the image
    """
    im_y, im_x = image.shape
    ker_y, ker_x = kernel.shape

    # We assume we're working with a square kernel for simplicity.
    assert ker_y == ker_x, "Please use a square kernel"

    out_y = im_y - ker_y + 1 # TODO: this will be the height of your output
    out_x = im_x - ker_x + 1 # TODO: this will be the width of your output

    convolved_image = np.zeros((out_y, out_x))
    for i in range(out_y):
        for j in range(out_x):
            # TODO now perform the actual convolution
            convolved_image[i][j] = np.sum(image[i:i+ker_y, j:j+ker_x]*kernel)

    return convolved_image

## Uncomment below to test your function
image = np.arange(9).reshape(3, 3)
print("Image:\n", image)
kernel = np.arange(4).reshape(2, 2)
print("Kernel:\n", kernel)
print("Convolved output:\n", convolution2d(image, kernel))