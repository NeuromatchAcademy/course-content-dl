## Get the samples
X_samples = X[0:5]
print("Sample input:", X_samples)

# Do a forward pass of the network
output = model.forward(X_samples)
print("Network output:", output)

# Predict the label of each point
y_predicted = model.predict(X_samples)
print("Predicted labels:", y_predicted)