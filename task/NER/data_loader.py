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
import json
from transformers import BertTokenizer
from config import Config


class DataGetter:
    """
DataLoader是pytorch.uu中的函数，用这个名字的话在其他类中调用很麻烦，所以我改名叫DataGetter了。
"""

    def __init__(self, path):
        # 数据集文件夹路径
        # self.dataset_dir = Config.dataset_base_path + '/' + dataset_name
        # # 读取数据集
        # self.train_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'train.json') #训练集
        # self.dev_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'dev.json') #验证集
        # self.test_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'test.json') #测试集
        self.path = path
        self.sentence = []
        self.label = []
        self.tokenized_sentence = []
        self.numeric_label = []
        self.tag2idx = {}

    # def load_tags(self):
    #     # 加载标签 比如CLUE-NER2020有10个标签类别，变成BIO标注后应该为21种标签，21种标签已保存在文件中，直接读就行
    #     tags = []
    #     tags_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'tags.txt')
    #     with open(tags_file, 'r') as f:
    #         for tag in f:
    #             tags.append(tag.strip())
    #     return tags

    def bio_converter(self):
        """
        Converting a normal font style file to BIO sequence labeling style
        :return:
        """
        with open(self.path, 'r') as fp:
            for line in fp.readlines():
                sentence = json.loads(line)  # read the json file line by line
                labels = sentence['label']
                text = sentence['text']
                text_list = list(text)
                target = []  # the 'target' list stores the output
                for i in range(len(text)):  # filling the output list with 'O' tag, which is the default tag
                    target.append('O')
                for key in labels:
                    entity_type = key
                    entity_dict = labels[key]  # e.g. {'叶老桂': [[9, 11]]}
                    for entity_name in entity_dict:
                        # searching for the label indexes in the target,
                        # replacing them with appropriate tag, the start tag is marked as 'B',
                        # 'I' tag is marked until the end of the label
                        entity_start_index = entity_dict[entity_name][0][0]
                        entity_end_index = entity_dict[entity_name][0][1]
                        entity_length = entity_end_index - entity_start_index + 1
                        target[entity_start_index] = 'B-' + str(entity_type)
                        if entity_length != 1:
                            for i in range(entity_start_index + 1, entity_end_index + 1):
                                target[i] = 'I-' + str(entity_type)
                self.label.append(target)
                self.sentence.append(text_list)
        fp.close()
        print("数据保存完毕")
        self.convert_labels_to_id()
        return self.label, self.sentence

    def count_labels(self):
        tag_values = []
        for label_list in self.label:
            for labels in label_list:
                tag_values.append(labels)
        tag_values = list(set(tag_values))
        self.tag2idx = {t: i for i, t in enumerate(tag_values)}

    def convert_labels_to_id(self):
        self.count_labels()
        self.numeric_label = [[self.tag2idx.get(l) for l in lab] for lab in self.label]

    def get_tag2idx(self):
        return self.tag2idx
