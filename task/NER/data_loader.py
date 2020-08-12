"""
data_loader用于读取所使用的公开数据集，并进行预处理

data_loader将公开数据集加上BIO标注，并将数据文本进行编码以便之后输入BERT
其他一些下游任务所需求的预处理工作也写在data_loader中
padding的代码，attention_mask确定最好也写在data_loader中

因此有些写的函数还没用上，但可能下游任务会用到，所以先写上了
目前所用数据集：https://github.com/CLUEbenchmark/CLUENER2020
"""
import random
import os
import torch
from transformers import BertTokenizer
from config import Config
 
 
class DataLoader:

    def __init__(self):
        
        # 数据集文件夹路径
        self.dataset_dir = dataset_base_path + '/' + dataset_name
        # 读取数据集
        self.train_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'train.json') #训练集
        self.dev_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'dev.json') #验证集
        self.test_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'test.json') #测试集
        # 读Bert词汇表
        self.tokenizer = BertTokenizer.from_pretrained(Config.model_vocab_path)


    def load_tags(self):
        # 加载标签 比如CLUE-NER2020有10个标签类别，变成BIO标注后应该为21种标签，21种标签已保存在文件中，直接读就行
        tags = []
        tags_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'tags.txt')
        with open(tags_file, 'r') as f:
            for tag in f:
                tags.append(tag.strip())
        return tags


    def set_BIO(self):
        # 给训练集和验证集打上BIO标注，但具体标注后是什么格式还需要结合下游任务模型
        save_dir_train = self.dataset_dir + '/train_BIO.json'
        save_dir_dev = self.dataset_dir + '/dev_BIO.json'

        # 打上标签后调用save_BIO_data存起来

    def save_BIO_data(self, data, save_dir):
        # 把打上BIO标注的dta存起来
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        


        print("数据保存完毕")


    def set_input_ids(self, sentences_file, result):
        # 读取文件中的句子变成BERT需求的编码
        

        # 可能需要先微调一下文件格式，下面读取句子的逻辑是每行为一个句子


        input_ids = []
        with open(sentences_file, 'r') as file: # 每行是一个句子
            for line in file:
                # 文本编码，变成BERT需求的input_ids
                input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
                input_ids = self.torch.tensor([input_ids])

        result['data'] = input_ids
        #result['size'] = len(input_ids)


    def data_iterator(self, data, shuffle=False):
        # padding 和 attention_mask相关的可写在这
        


if __name__ == '__main__':

    data_loader = DataLoader()



