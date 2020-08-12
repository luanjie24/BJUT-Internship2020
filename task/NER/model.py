
"""
model.py用于定义模型类
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn import CrossEntropyLoss

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()   
        # BERT层
        self.config = BertConfig.from_pretrained(Config.model_config_path)
        self.bert = BertModel.from_pretrained(Config.model_path, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True #微调时是否调BERT，True的话就调

        # BiLSTM层


        # CRF层


        # 计算损失
        self.loss_fct = CrossEntropyLoss() #用交叉熵不知道合不合适

    def forward(self, input_tensor, attention_mask=None):
        #attention_mask用于微调BERT，是对padding部分进行mask
        embedding_output, _ = self.bert(input_ids, attention_mask=attention_mask)  #shape:(batch_size, sequence_length, 768)
        
        