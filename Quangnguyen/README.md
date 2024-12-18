# Quangnguyen711/solidity_re_entrancy_dataset

   ## Overview
                The Solidity Re-Entrancy Dataset is a comprehensive collection of Solidity smart contracts designed to highlight re-entrancy vulnerabilities. Re-entrancy is one of the most common and critical vulnerabilities in Ethereum smart contracts, where malicious external calls can exploit contract logic and cause unintended behaviors.

                This dataset serves as a resource for:

                         Smart contract security analysis
                         Machine learning model training for vulnerability detection
                         Educational purposes for developers and researchers


## Dataset Structure :
 | func | label	| index_level_0  |
 |------| ------| -----------|
 | function donateToken(address _token, uint256 _amount) external {} | 1 |	3826   |                   
 | function addOperator(address newOperator) public {}	 |0 |	13932 |


  # Applications
                This dataset can be used for the following applications:

### Smart Contract Vulnerability Detection:
                Train machine learning models to classify functions as vulnerable (1) or safe (0).

### Static Analysis Tools:
                Build automated tools to analyze Solidity code for re-entrancy vulnerabilities.

### Security Education:
                Teach developers to identify and mitigate re-entrancy issues through real-world examples.

### AI Research:
                Develop and benchmark deep learning models, such as transformers, for code vulnerability detection.

### Benchmarking:
                Compare the effectiveness of existing tools for smart contract security against this labeled dataset.


## Dataset  
 ### 1. Clone the Dataset  :

             from datasets import load_dataset
             dataset = load_dataset("Quangnguyen711/solidity_re_entrancy_dataset")
             print(dataset["train"][0])
            
 ### 2. Dataset Exploration :

              print(dataset)
              print(dataset["train"].to_pandas().head())


## Use the Dataset:

### Preprocess the Dataset for Machine Learning

from transformers import T5Tokenizer
from datasets import Dataset
tokenizer = T5Tokenizer.from_pretrained("t5-small")


def preprocess_function(examples):
    inputs = [f"Analyze this Solidity function: {func}" for func in examples["func"]]
    targets = ["vulnerable" if label == 1 else "safe" for label in examples["label"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)
print(tokenized_dataset)


### Train a Model to Detect Vulnerabilities


You can fine-tune a T5 model to classify Solidity functions as vulnerable or safe.

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
model = T5ForConditionalGeneration.from_pretrained("t5-small")


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"].select(range(100)), # Example small eval set
)


trainer.train()


## Dataset Statistics

+ Total Functions: 10,000+
+ Vulnerable Samples: Approximately 50%
+ Non-Vulnerable Samples: Approximately 50%
The dataset is balanced, ensuring equal representation of vulnerable and non-vulnerable functions.


## Dependencies

pip install transformers datasets pandas

## License

This dataset is released under the MIT License.

       


