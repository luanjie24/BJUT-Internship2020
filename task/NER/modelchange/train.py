"""
train.py用于训练集的训练

"""
from torch import nn
import torch
from data_getter import DataGetter
from model import Model
from config import Config

# -*- coding:utf-8 -*-
import pickle
import sys
import yaml
import torch.optim as optim
#from utils import f1_score, get_tags, format_result
from custom_dataset import CustomData

class Train(object):
    
    def __init__(self, entry="train"):
        self.load_config()
        self.__init_model(entry)

    def __init_model(self):
        self.train_manager = DataGetter('D://nlp work//gwx//task//NER//dataset//CLUE-NER2020//train.json')
        #self.total_size = len(self.train_manager.sentence)
        data = {
            "batch_size": 20,
            #"input_size": self.train_manager.input_size,
            "vocab": self.train_manager.vocab,
            "tag_map": self.train_manager.get_tag2idx,
        }
        self.save_params(data)
        dev_manager = DataGetter('D://nlp work//gwx//task//NER//dataset//CLUE-NER2020//train.json')
        #self.dev_batch = dev_manager.iteration()

        self.model = BiLSTMCRF(
            tag_map=self.train_manager.tag_map,
            batch_size=self.batch_size,
            vocab_size=Config.max_len,
            dropout=self.dropout,
            embedding_dim=self.embedding_size,
            hidden_dim=self.hidden_size,
        )
        self.restore_model()

    def load_config(self):
        try:
            fopen = open("config.py",'r',encoding='UTF-8')
            fopen.close()
        except Exception as error:
            print("Load config failed, using default config {}".format(error))
            fopen = open("config.py", "w", encoding='UTF-8')
            config = {
                "embedding_size": 3,
                "hidden_size": 128,
                "batch_size": 20,
                "dropout":0.5,
            }
            fopen.close()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = "models/"
        self.tags = train_manager.label
        self.dropout = config.get("dropout")

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    def save_params(self, data):
        with open("models/data.pkl", "wb") as fopen:
            pickle.dump(data, fopen)

    def load_params(self):
        with open("models/data.pkl", "rb") as fopen:
            data_map = pickle.load(fopen)
        return data_map

    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(ner_model.parameters(), lr=0.01)
        for epoch in range(1):
            index = 0
            for batch in self.train_manager.get_batch():
                index += 1
                self.model.zero_grad()
                
                train_labels, train_sentences = train_manager.bio_converter()
                input_tensor=CustomData.__getitem__(index)

                sentences_tensor = torch.tensor(train_sentences, dtype=torch.long)#转换为张量
                tags_tensor = torch.tensor(train_labels, dtype=torch.long)
                length_tensor = torch.tensor(len(self.train_manager.sentence), dtype=torch.long)

                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)
                progress = ("█"*int(index * 25 / self.total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                        epoch, progress, index, self.total_size, loss.cpu().tolist()[0]
                    )
                )
                self.evaluate(input_tensor)
                print("-"*50)
                loss.backward()
                optimizer.step()
                torch.save(self.model.state_dict(), self.model_path+'params.pkl')

    def evaluate(self,input_tensor):
        train_labels, train_sentences = train_manager.bio_converter()
        length=len(self.train_manager.sentence)
        _, paths = self.model(input_tensor, train_sentences)
        print("\teval")
        for label in self.labels:
            f1_score(labels, paths, label, self.model.tag_map)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\ttrain\n")
        exit()  
    cn = Train("train")
    cn.train()