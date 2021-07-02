def visualize_data(dataloader):

  for idx, (data,label) in enumerate(dataloader):
    plt.figure(idx)
    # Choose the datapoint you would like to visualize
    index = 22

    # choose that datapoint using index and permute the dimensions
    # and bring the pixel values between [0,1]
    data = data[index].permute(1, 2, 0) * \
           torch.tensor([0.5, 0.5, 0.5]) + \
           torch.tensor([0.5, 0.5, 0.5])

    # Convert the torch tensor into numpy
    data = data.numpy()

    plt.imshow(data)
    plt.axis(False)
    image_class = classes[label[index].item()]
    print(f'The image belongs to : {image_class}')

  plt.show()


## uncomment to run the function
with plt.xkcd():
  visualize_data(rand_train_loader)