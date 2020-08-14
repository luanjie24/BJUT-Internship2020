"""
model.py用于定义模型类
"""
import torch
from transformers import BertForTokenClassification
from config import Config


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BertClass, self).__init__()
        self.l1 = BertForTokenClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=21,
            output_attention=False,
            output_hidden_states=False
        )

    def forward(self, ids, masks, labels):
        output = self.l1(ids, masks, labels=labels)
        return output
