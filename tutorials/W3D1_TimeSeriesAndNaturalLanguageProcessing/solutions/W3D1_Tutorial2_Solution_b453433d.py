
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)