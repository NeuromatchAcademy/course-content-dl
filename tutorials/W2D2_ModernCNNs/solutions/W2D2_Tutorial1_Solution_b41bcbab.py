def predict_top5(images):
  ''' Args:
          images: torch Tensor with dimensionality B x C x H x W.
      Output:
          top5_probs: torch Tensor (B, 5) with top 5 class probabilities
          top5_names: list of top 5 class names (B, 5)
  '''
  B = images.size(0)
  with torch.no_grad():
    # run images through model
    images = images.to(device)
    output = resnet(images)
    probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()

  _, top5_idcs = output.topk(5, 1, True, True)
  top5_idcs = top5_idcs.t().cpu().numpy()

  top5_probs = probs[torch.arange(B), top5_idcs]

  top5_names = []
  for b in range(B):
    temp = [dict_map[key].split(',')[0] for key in top5_idcs[:,b]]
    top5_names.append(temp)

  return top5_names, top5_probs