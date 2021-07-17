"""
Create a TxT matrix with values between 0 and 1 to
be used to compute the metrics.
"""
T = 5  # number of tasks
result_matrix = np.random.uniform(low=0, high=1, size=(T, T))  # here put a TxT matrix with values in [0, 1]