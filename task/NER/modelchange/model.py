
"""
model.py用于定义模型类
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from data_getter import DataGetter
from custom_dataset import CustomData
from transformers import BertModel, BertConfig
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

START_TAG = "START"
STOP_TAG = "STOP"

class Model(nn.Module):
    def __init__(self,
            tag_map=self.DataGetter.tag2idx,#使用BIOES做序列标记，此为映射
            batch_size=20,#表示一次输入多少个数据，默认放在第二维度，eg：input(_,batch_size,_)
            vocab_size=20,#词汇/////
            hidden_dim=128,#隐藏状态的大小; 每个LSTM单元在每个时间步产生的输出数（我理解的是lstm的神经元个数）
            dropout=0.5,#在除最后一个 RNN 层外的其他 RNN 层后面加 dropout 层，数值表示加入的概率，1.0表示100%加入
            embedding_dim=100#embedding向量维度///
            ):
        super(Model, self).__init__()   
        # BERT层
        self.config = BertConfig.from_pretrained(Config.model_config_path)
        self.bert = BertModel.from_pretrained(Config.model_path, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True #微调时是否调BERT，True的话就调

        # BiLSTM层
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.tag_size = len(tag_map)#tag_size=6
        self.tag_map = tag_map
        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size)
        )
        self.transitions.data[:, START_TAG] = -1000.
        self.transitions.data[STOP_TAG, :] = -1000.

        #(使用bert输出的embedding，去掉下面两句的注释，做适当修改，详情参考https://blog.csdn.net/appleml/article/details/78595724?%3E)定义词向量大小，参数一为单词个数，二为单词长度
        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        #pretrained_weight = np.array(pretrained_weight)
        #self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))
        
        #self.hidden_dim // 2隐藏层状态的维数,LSTM 堆叠的层数num_layers=1,bidirectional: 是双向 RNN,batch_first=True输入输出的第一维为 batch_size
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim // 2,num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)#线性层将隐状态空间映射到标注空间
        self.hidden = self.init_hidden()#重新初始化隐藏层数据，避免受之前运行代码的干扰,如果不重新初始化，会有报错。

        # CRF层

        # 计算损失
        self.loss_fct = CrossEntropyLoss() #用交叉熵不知道合不合适

    def init_hidden(self):#开始时刻, 没有隐状态 #从标准正态分布（均值为0，方差为1）中抽取的一组随机数。返回一个张量.输出张量的形状为2，
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def __get_lstm_features(self, sentence):#经过了embedding，lstm，linear层，是根据LSTM算出的一个发射矩阵
        self.hidden = self.init_hidden()
        length = sentence.shape[1]
        embeddings = self.word_embeddings(sentence).view(self.batch_size, length, self.embedding_dim)

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        logits = self.hidden2tag(lstm_out)#发射矩阵，表示发射概率，表示经过LSTM后sentence的每个word对应的每个labels的得分
        return logits
    #求正确路径pair: feats->tags 的分值
    def real_path_score_(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)#score=0
        tags = torch.cat([torch.tensor([self.tag_map[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i], tags[i+1]] + feat[tags[i + 1]]
        score = score + self.transitions[tags[-1], self.tag_map[STOP_TAG]]
        return score
    #CRF的分子对数
    def real_path_score(self, logits, label):
        '''
        caculate real path score  
        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * len_sent]

        Score = Emission_Score + Transition_Score  
        Emission_Score = logits(0, label[START]) + logits(1, label[1]) + ... + logits(n, label[STOP])  
        Transition_Score = Trans(label[START], label[1]) + Trans(label[1], label[2]) + ... + Trans(label[n-1], label[STOP])  
        '''
        score = torch.zeros(1)
        label = torch.cat([torch.tensor([4], dtype=torch.long), label])
        for index, logit in enumerate(logits):#沿途累加每一帧的转移和发射
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score
        score += self.transitions[label[-1], 5]
        return score
    #CRF的分母，不限定隐状态路径，求出所有路径对应分值之和
    def total_score(self, logits, label):
        """
        caculate total score

        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * tag_size]

        SCORE = log(e^S1 + e^S2 + ... + e^SN)
        """
        obs = []
        previous = torch.full((1, self.tag_size), 0.0)
        for index in range(len(logits)): 
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, 5]
        # caculate total_scores
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores
    #求loss
    def neg_log_likelihood(self, sentences, tags, length):
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)#经过了LSTM+Linear矩阵后的输出，求出了每一帧对应到每种tag的"发射【分值】矩阵"，之后作为CRF的输入
        real_path_score = torch.zeros(1)
        total_score = torch.zeros(1)
        for logit, tag, leng in zip(logits, tags, length):
            logit = logit[:leng]
            tag = tag[:leng]
            real_path_score += self.real_path_score(logit, tag)#所有正确路径的分数和
            total_score += self.total_score(logit, tag)#不限定隐状态路径，求出所有路径对应分值之和
        # print("total score ", total_score)
        # print("real score ", real_path_score)
        return total_score - real_path_score #根据CRF的公式定义，两者之间的差值作为loss，之后直接根据这个差值，反向传播
    #lstm的前向函数，输入sentences序列经过LSTM，得到对应的发射矩阵。推断逻辑很直观，就是过一遍LSTM拿到每一帧的发射状态分布；然后跑viterbi解码得出最优路径和分值。

    #求最优路径分值 和 最优路径

    def __viterbi_decode(self, logits):
        backpointers = [] ## 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是神马状态
        trellis = torch.zeros(logits.size())
        backpointers = torch.zeros(logits.size(), dtype=torch.long)
        
        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.numpy()
        for bp in reversed(backpointers[1:]):## 从[1:]开始，去掉开头的 START_TAG
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi## 返回最优路径分值 和 最优路径


    #lstm的前向函数，输入sentences序列经过LSTM，得到对应的发射矩阵
    def forward(self, input_tensor, attention_mask=None, sentences,lengths=None):
        #attention_mask用于微调BERT，是对padding部分进行mask
        embedding_output, _ = self.bert(input_ids, attention_mask=attention_mask)  #shape:(batch_size, sequence_length, 768)、
        print("embedding_output.shape:"embedding_output.shape)

        #------------------------------------------------------------------
        sentences = torch.tensor(sentences, dtype=torch.long)
        if not lengths:
            lengths = [i.size(-1) for i in sentences]
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences)# 求出每一帧的发射矩阵
        #求所有路径的得分指数和的对数
        scores = []
        paths = []
        for logit, leng in zip(logits, lengths):
            logit = logit[:leng]
            score, path = self.__viterbi_decode(logit)#采用已经训好的CRF层, 做维特比解码, 得到最优路径及其分数
            scores.append(score)
            paths.append(path)

        return scores, paths
    
    #以下在usemodel是用到
    #utils.py 实体的标记 
    def format_result(result, text, tag): 
        entities = [] 
        for i in result: 
            begin, end = i 
            entities.append({ 
                "start":begin, 
                "stop":end + 1, 
                "word":text[begin:end+1],
                "type":tag
            }) 
        return entities
    #utils.py 标记tag
    def get_tags(path, tag, tag_map):
        begin_tag = tag_map.get("B-" + tag)
        mid_tag = tag_map.get("I-" + tag)
        end_tag = tag_map.get("E-" + tag)
        single_tag = tag_map.get("S")
        o_tag = tag_map.get("O")
        begin = -1
        end = 0
        tags = []
        last_tag = 0
        for index, tag in enumerate(path):
            if tag == begin_tag and index == 0:
                begin = 0
            elif tag == begin_tag:
                begin = index
            elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
                end = index
                tags.append([begin, end])
            elif tag == o_tag or tag == single_tag:
                begin = -1
            last_tag = tag
        return tags
    #对模型的评价
    def f1_score(tar_path, pre_path, tag, tag_map):
        origin = 0.
        found = 0.
        right = 0.
        for fetch in zip(tar_path, pre_path):
            tar, pre = fetch
            tar_tags = get_tags(tar, tag, tag_map)
            pre_tags = get_tags(pre, tag, tag_map)

            origin += len(tar_tags)
            found += len(pre_tags)

            for p_tag in pre_tags:
                if p_tag in tar_tags:
                    right += 1

        recall = 0. if origin == 0 else (right / origin)
        precision = 0. if found == 0 else (right / found)
        f1 = 0. if recall+precision == 0 else (2*precision*recall)/(precision + recall)
        print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(tag, recall, precision, f1))
        return recall, precision, f1



        
        