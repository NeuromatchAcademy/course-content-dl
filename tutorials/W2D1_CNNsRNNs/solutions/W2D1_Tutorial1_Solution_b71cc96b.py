def solution_conv():
    solution = np.array([[63, 63*3], [0,  63]]).astype(np.uint8)

    return solution


original = np.array([
                     [0, 252, 252],
                     [0,   0, 252],
                     [0,   0,   0]]).astype(np.uint8)
# region collapse
# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
# endregion

img_original = Image.fromarray(original, 'L')
plt.figure(figsize=(4, 4))
plt.imshow(img_original.resize((120,120), Image.NEAREST), cmap='gray')
plt.title("Original image")
plt.axis('off')

solution = solution_conv()
if solution.shape[0] > 1:
  img_solution = Image.fromarray(solution, 'L')
  plt.figure(figsize=(3, 3))
  plt.imshow(img_solution.resize((80, 80), Image.NEAREST), cmap='gray')
  plt.title("Convolution result")
  plt.axis('off')