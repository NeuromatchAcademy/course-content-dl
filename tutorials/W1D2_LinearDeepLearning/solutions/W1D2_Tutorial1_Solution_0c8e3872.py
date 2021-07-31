def fun_z(x, y):
  """Function sin(x^2 + y^2)

  Args:
    x (float, np.ndarray): variable x
    y (float, np.ndarray): variable y

  Return:
    z (float, np.ndarray): sin(x^2 + y^2)
  """
  z = np.sin(x**2 + y**2)
  return z


def fun_dz(x, y):
  """Function sin(x^2 + y^2)

  Args:
    x (float, np.ndarray): variable x
    y (float, np.ndarray): variable y

  Return:
    (tuple): gradient vector for sin(x^2 + y^2)
  """
  dz_dx = 2 * x * np.cos(x**2 + y**2)
  dz_dy = 2 * y * np.cos(x**2 + y**2)
  return (dz_dx, dz_dy)

#add event to airtable
atform.add_event('Coding Exercise 1.1: Gradient Vector')

## Uncomment to run
with plt.xkcd():
  ex1_plot(fun_z, fun_dz)