import torch

class Config:
    device = torch.device('cuda: 3' if torch.cuda.is_available() else 'cpu') #看有没有cuda，没有就只能用cpu了

    # 数据集的路径 
    # 访问格式:  file = os.path.join(Config.dataset_base_path, Config.dataset_name, 文件名)
    dataset_base_path = 'project/dataset' 
    dataset_name = 'CLUE-NER2020' # 目前用到的数据集名称（文件夹名称），以后用别的直接在这里改

    # BERT相关路径
    model_path = 'project/bert_pretrainpytorch_model.bin'
    model_vocab_path = 'project/bert_pretrain/bert-base-chinese-vocab.txt'
    model_config_path = 'project/bert_pretrain/bert_config.json'

    # 一些超参数
    batch_size = 20
    learning_rate = 0.001
    max_len = 180

    epoch_num = 20
    num_tag = 7
    clip_grad = 2
    
    embedding_size: 100
    hidden_size: 128
    dropout:0.5