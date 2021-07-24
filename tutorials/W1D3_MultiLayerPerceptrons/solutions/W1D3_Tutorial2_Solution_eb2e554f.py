
"""
No, we don't see low-level features in the first layer (like edges or face parts).
This is because the MLP model does not have a preference (bias or prior)
for hierarchical feature maps; hence it won't learn it by default.
""";