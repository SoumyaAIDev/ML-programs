from transformers import T5Tokenizer
from datasets import Dataset

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess_dataset(dataset):
    def preprocess_function(example):
        inputs = f"Detect vulnerability: {example['func']}"
        targets = "vulnerable" if example['label'] == 1 else "safe"
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    return dataset.map(preprocess_function)

if __name__ == "__main__":
    dataset = load_solidity_dataset()
    tokenized_dataset = preprocess_dataset(dataset["train"])
    print(tokenized_dataset)
