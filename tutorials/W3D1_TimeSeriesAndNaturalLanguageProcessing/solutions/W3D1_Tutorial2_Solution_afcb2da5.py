def evaluate(net, all_characters, prime_str, predict_len):
  hidden = net.init_hidden()
  predicted = prime_str

  # "Building up" the hidden state
  for p in range(len(prime_str) - 1):
    inp = char_tensor(prime_str[p])
    _, hidden = net(inp, hidden)

  # Tensorize of the last character
  inp = char_tensor(prime_str[-1])

  # For every index to predict
  for p in range(predict_len):
    # Pass the inputs to the model
    output, hidden = net(inp, hidden)

    # Pick the character with the highest probability
    top_i = torch.argmax(torch.softmax(output, dim=1))

    # Add predicted character to string and use as next input
    predicted_char = all_characters[top_i]
    predicted += predicted_char
    inp = char_tensor(predicted_char)

  return predicted


## Uncomment to run
sampleDecoder = GenerationRNN(27, 100, 27, 1)
text = evaluate(sampleDecoder, all_characters, 'hi', 10)
if text.startswith('hi') and len(text) == 12:
  print('Success!')
else:
  print('Need to change.')

# add event to airtable
atform.add_event('Coding Exercise 3: Implement Text generation')