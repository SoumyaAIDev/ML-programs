from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk

def train_model():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    dataset = load_from_disk("data/processed/tokenized_dataset")

    training_args = TrainingArguments(
        output_dir="./models/t5-finetuned",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./output/logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    train_model()
