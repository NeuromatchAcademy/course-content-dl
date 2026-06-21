def generate_text(model, input_prompt):
  # Number of tokens to generate
  num_tokens = 100

  # Move the model to the CPU for inference
  model.to("cpu")

  # Print input prompt
  print(f'Input prompt: \n{input_prompt}')

  # Encode the input prompt
  # https://huggingface.co/docs/transformers/en/main_classes/tokenizer
  input_tokens = tokenizer.encode(input_prompt)

  # Turn off storing gradients
  with torch.no_grad():
    # Keep iterating until num_tokens are generated
    for tkn_idx in tqdm(range(num_tokens)):
      # Forward pass through the model
      output = model(torch.LongTensor(input_tokens).unsqueeze(0))
      logits = output.logits[0, -1, :]
      # Convert into probabilities
      probs = nn.functional.softmax(logits, dim=-1)
      # Get the index of top token
      top_token = torch.argmax(probs).item()
      # Append the token into the input sequence
      input_tokens.append(top_token)

  # Decode and print the generated text
  # https://huggingface.co/docs/transformers/en/main_classes/tokenizer
  decoded_text = tokenizer.decode(input_tokens)
  return decoded_text

print(f'Generated text: \n{generate_text(model, input_prompt)}')