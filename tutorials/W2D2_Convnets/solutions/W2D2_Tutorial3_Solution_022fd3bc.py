def predict_top5(images, device, seed):
  """
  Function to predict top 5 classes

  Args:
    images: torch.tensor
      Image data with dimensionality B x C x H x W batch size x number of channels x height x width)
    device: STRING
      `cuda` if GPU is available, else `cpu`.

  Output:
    top5_probs: torch.tensor
      Tensor(B, 5) with top 5 class probabilities
    top5_names: list
      List of top 5 class names (B, 5)
  """
  set_seed(seed=seed)

  B = images.size(0)
  with torch.no_grad():
    # Run images through model
    images = images.to(device)
    output = resnet(images)
    # The model output is unnormalized. To get probabilities, run a softmax on it.
    probs = torch.nn.functional.softmax(output, dim=1)
    # Fetch output from GPU and convert to numpy array
    probs = probs.cpu().numpy()

  # Get top 5 predictions
  _, top5_idcs = output.topk(5, 1, True, True)
  top5_idcs = top5_idcs.t().cpu().numpy()
  top5_probs = probs[torch.arange(B), top5_idcs]

  # Convert indices to class names
  top5_names = []
  for b in range(B):
    temp = [dict_map[key].split(',')[0] for key in top5_idcs[:, b]]
    top5_names.append(temp)

  return top5_names, top5_probs


# Get batch of images
dataiter = iter(imagenette_val_loader)
images, labels = next(dataiter)

## Uncomment to test your function and retrieve top 5 predictions
top5_names, top5_probs = predict_top5(images, DEVICE, SEED)
print(top5_names[1])