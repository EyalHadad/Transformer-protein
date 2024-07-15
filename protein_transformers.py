import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import PreTrainedTokenizerFast, RobertaTokenizerFast, RobertaForSequenceClassification
from datasets import Dataset
import torch

# Load training and test data
train_df = pd.read_csv('train_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

# Map labels to integers
label_map = {'ribosomal': 0, 'kinase': 1, 'ligase': 2}
train_df['label'] = train_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

# Convert DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Model 1: English base model (e.g., BERT)
base_model_name = 'bert-base-uncased'
tokenizer1 = AutoTokenizer.from_pretrained(base_model_name)
model1 = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=3)

def tokenize_function(examples):
    return tokenizer1(examples['sequence'], padding='max_length', truncation=True)

train_dataset1 = train_dataset.map(tokenize_function, batched=True)
test_dataset1 = test_dataset.map(tokenize_function, batched=True)

train_dataset1.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset1.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
)

trainer1 = Trainer(
    model=model1,
    args=training_args,
    train_dataset=train_dataset1,
    eval_dataset=test_dataset1,
    tokenizer=tokenizer1,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer1)
)

trainer1.train()
trainer1.save_model('./model1')

# Evaluate model 1
results1 = trainer1.evaluate()
with open('results1.txt', 'w') as f:
    f.write(str(results1))

# Model 2: Train tokenizer from scratch
# Create custom tokenizer
tokenizer2 = RobertaTokenizerFast.from_pretrained('roberta-base')
tokenizer2.add_special_tokens({'pad_token': '[PAD]'})

# Train a new tokenizer
def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]['sequence']

tokenizer2.train_new_from_iterator(batch_iterator(train_dataset), vocab_size=52000)
model2 = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
model2.resize_token_embeddings(len(tokenizer2))

# Tokenize using the new tokenizer
def tokenize_function2(examples):
    return tokenizer2(examples['sequence'], padding='max_length', truncation=True)

train_dataset2 = train_dataset.map(tokenize_function2, batched=True)
test_dataset2 = test_dataset.map(tokenize_function2, batched=True)

train_dataset2.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset2.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

trainer2 = Trainer(
    model=model2,
    args=training_args,
    train_dataset=train_dataset2,
    eval_dataset=test_dataset2,
    tokenizer=tokenizer2,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer2)
)

trainer2.train()
trainer2.save_model('./model2')

# Evaluate model 2
results2 = trainer2.evaluate()
with open('results2.txt', 'w') as f:
    f.write(str(results2))

print("Training and evaluation complete. Models and results saved.")
