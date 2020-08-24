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
from config import Config


class DataGetter:
    def __init__(self, path):
        # 数据集文件夹路径
        # self.dataset_dir = Config.dataset_base_path + '/' + dataset_name
        # # 读取数据集
        # self.train_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'train.json') #训练集
        # self.dev_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'dev.json') #验证集
        # self.test_file = os.path.join(Config.dataset_base_path, Config.dataset_name, 'test.json') #测试集
        self.path = path # 数据集路径
        self.sentence = [] #数据集中的所有句子 [['张', '三', '被', '抓'], ['...']]
        self.label = [] #数据集中的所有的BIO标注 [['B-per', 'I-per', 'o', 'O'], ['...']]
        self.numeric_label = [] #所有句子的所有标签编号 [[1, 2, 21，21], [...]]
        self.tag2idx = {} #标签：标签id的字典 {'I-name': 0, 'B-address': 1, 'I-game': 2 ...}



    def bio_converter(self):
        # Converting a normal font style file to BIO sequence labeling style
        with open(self.path, 'r', encoding="utf-8") as fp:
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

    def convert_labels_to_id(self):
        self.count_labels()
        self.numeric_label = [[self.tag2idx.get(l) for l in lab] for lab in self.label]

    def count_labels(self):
      # 通过遍历整个label数组得到该文件中一共有多少种label组合。其中Label组合是指形如i-geo 或者 b-per这类的BIO标注与tag的组合
        tag_values = []
        for label_list in self.label:
            for labels in label_list:
                tag_values.append(labels)
        tag_values = list(set(tag_values))
        self.tag2idx = {t: i for i, t in enumerate(tag_values)}
        self.tag2idx['START']=21
        self.tag2idx['STOP']=22

    def get_tag2idx(self):
      # 得到储存有所有tag与index的字典tag2idx
      return self.tag2idx

    def get_numeric_labels(self):
      return self.numeric_label



if __name__ == "__main__":
    train_data_loader = DataGetter('D://nlp work//gwx//task//NER//dataset//CLUE-NER2020//train.json')
    train_labels, train_sentences = train_data_loader.bio_converter()
    train_labels = train_data_loader.get_numeric_labels()
    # print("sentence)", train_data_loader.sentence)
    # print("label", train_data_loader.label)
    # print("numeric_label" ,train_data_loader.numeric_label)
    # print("label" ,train_data_loader.label)
    print("tag2idx", train_data_loader.tag2idx) 