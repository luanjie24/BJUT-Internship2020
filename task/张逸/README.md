# 使用 gensim 训练中文词向量


## 套件需求

* jieba
```
pip3 install jieba
```
* gensim
```
pip3 install -U gensim
```
* [OpenCC](https://github.com/BYVoid/OpenCC) (简繁转换组件)

## 训练流程

1.取得[中文维基数据](https://dumps.wikimedia.org/zhwiki/20160820/zhwiki-20160820-pages-articles.xml.bz2)，本次實驗是採用 2016/8/20 的資料。

*前往[wiki百科:资料库下载](https://dumps.wikimedia.org/zhwiki/)( 挑选以`pages-articles.xml.bz2`为结尾的档案 )*

2.下载文本集后使用`wiki_to_txt.py`从 xml 中提取出wiki文章

```
python3 wiki_to_txt.py zhwiki-20160820-pages-articles.xml.bz2
```
*本次训练数据集采用发布的最新文本集“zhwiki-latest-pages-articles.xml.bz2”*

3.使用 OpenCC 将wiki文章转换为简体中文
```
opencc -i wiki_texts.txt -o wiki_zh_tw.txt -c s2tw.json
```
4.使用`jieba` 对文本进行分词，并删除停用词
```
python3 segment.py
```
5.使用`gensim` 的 word2vec 模型进行训练
```
python3 train.py
```
6.测试训练出的模型
```
python3 demo.py
```
