import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from Process import *
from transformers import (RobertaTokenizer,
                          RobertaForSequenceClassification,
                          Trainer,
                          TrainingArguments)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support,
                             confusion_matrix)

from transformers import AutoTokenizer, AutoModelForSequenceClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thay RobertaTokenizer bằng AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('mr4/phobert-base-vi-sentiment-analysis')
model = AutoModelForSequenceClassification.from_pretrained(
    'mr4/phobert-base-vi-sentiment-analysis',
    num_labels=2,
    ignore_mismatched_sizes=True).to(device) 

train = pd.read_csv(r'google-review-scraper-main\datareview.csv')
train['Processed Review Text'] = train['Review Text'].apply(preprocess_text)
train['Rating'] = train['Rating'].apply(lambda x: int(x.replace(' sao', '')))
threshold = 3
train['target'] = (train['Rating'] > threshold).astype(int)

X = train['Processed Review Text']
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset class
class FPTtraindt(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    from transformers import EarlyStoppingCallback, TrainingArguments, Trainer

# Tokenize the data
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

# Create datasets
train_dataset = FPTtraindt(train_encodings, y_train.tolist())
test_dataset = FPTtraindt(test_encodings, y_test.tolist())

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=250,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to='none',
    no_cuda=False 
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()
