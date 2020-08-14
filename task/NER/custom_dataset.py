import torch
from data_loader import DataGetter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from config import Config


class CustomData:
    def __init__(self, tokenizer, sentences, labels):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels = labels
        self.len = len(sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        inputs = tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=Config.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label.extend([4] * Config.max_len)  # O
        label = label[:Config.max_len]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return self.len