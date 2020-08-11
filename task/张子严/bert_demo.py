import torch
from transformers import BertForTokenClassification, BertTokenizer, BertConfig
import pandas as pd
import numpy as np
from torch.utils.data import *

# Defining some key variables
max_len = 150
train_batch_size = 32
test_batch_size = 16
epochs = 5
learning_rate = 2e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
importing and pre-processing the data (missing part)
'''
# dataFrame = pd.read_csv('train.csv', error_bad_lines=False)

# preparing the dataSet and the dataLoader
class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels = labels
        self.max_len = max_len
        self.len = len(sentences)

    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(sentence, None, add_special_tokens=True,
                                            max_length=self.max_len, pad_to_max_length=True,
                                            return_token_type_ids=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = self.labels[index]
        label.extend([4] * 200)
        label = label[:150]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return self.len


# creating train data set and test data set
# creating the nueral  network for fine tuning
class BertClass(torch.nn.Module):
    def __init__(self):
        super(BertClass, self).__init__()
        self.l1 = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=18)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 150)

    def forward(self, ids, mask, labels):
        output_1 = self.l1(ids, mask, labels=labels)
        output_2 = self.l2(output_1[0])
        output = self.l3(output_2)
        return output


training_loader = DataLoader()  # Need to add parameters after we sticking the data set
model = BertClass()
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


# fine tuning the model
def train(epoch):
    model.train()
    for i, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['tags'].to(device, dtype=torch.long)

        loss = model(ids, mask, labels=targets)[0]

        if i % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()  # discarding the grad before go backward
        loss.backward()
        optimizer.step()


for epoch in range(5):
    train(epoch)

'''
Validating the model thru f1 score (also a missing part)
'''
