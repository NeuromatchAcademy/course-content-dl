
from torchvision.transforms import Compose, Grayscale

# TODO Load the CIFAR10 data using a transform that converts the images to grayscale tensors
data = datasets.CIFAR10(
    root="data",
    download=True,
    transform=Compose([ToTensor(),Grayscale()])
)


# Display a random grayscale image
image, label = data[random.randint(0, len(data))]
plt.imshow(image.squeeze(), cmap="gray")