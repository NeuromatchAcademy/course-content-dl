from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import numpy as np
from evaluate import load
metric = load("accuracy")

# Trainer:
training_args = TrainingArguments(
    output_dir="./codeparrot",
    max_steps=100,
    per_device_train_batch_size=1,
)

tokenizer.pad_token = tokenizer.eos_token

encoded_dataset = dataset.map(
    lambda x: tokenizer(x["code"], truncation=True, padding="max_length"),
    batched=True,
    remove_columns=["code"],
)


# Metrics for loss:
def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=-1)
  return metric.compute(predictions=predictions, references=labels)


# Data collator:
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)