# -*- coding: utf-8 -*-

import logging
import pickle

from gensim.corpora import Dictionary
from gensim.models import word2vec
def create_dictionaries(p_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.wv.vocab.keys(), allow_update=True)

    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: p_model[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=250)
    #model = word2vec.Word2Vec.load("word2vec.model")  #加载模型
    model.train(sentences, total_examples=model.corpus_count, epochs=10)

    #保存模型，供日后使用
    model.save("word2vec.model")
    #保存特征矩阵   (.format文件和.bin文件皆可)
    model.wv.save_word2vec_format('./word2vec_200.bin', binary=False)
    ######参考其他项目代码，生成字典，读取内容后并没有明白作用。。
    index_dict, word_vectors = create_dictionaries(model)
    pkl_name = input("请输入保存的pkl文件名...\n")
    output = open(pkl_name + u".pkl", 'wb')
    pickle.dump(index_dict, output)  # 索引字典
    pickle.dump(word_vectors, output)  # 词向量字典
    output.close()
    ######

    #模型读取方式
    # model = word2vec.Word2Vec.load("your_model_name")

if __name__ == "__main__":
    main()
