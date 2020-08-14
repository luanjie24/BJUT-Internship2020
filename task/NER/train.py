"""
train.py用于训练集的训练

"""
from torch import nn
import torch
from data_loader import DataLoader
from model import Model
from config import Config


def train(epoch):
    model.train()
    for step, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['tags'].to(device, dtype=torch.long)

        loss = model(ids, mask, labels=targets)

        if step % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_data_loader = DataGetter('/Users/ziyanzhang/PycharmProjects/train.json')
train_labels, train_sentences = train_data_loader.bio_converter()
tag2idx = train_data_loader.get_tag2idx()
training_set = CustomData(tokenizer, train_sentences, train_data_loader)
train_params = {'batch_size': Config.batch_size,
                'shuffle': True,
                'num_workers': 0
                }
training_loader = DataLoader(training_set, **train_params)

model.cuda()

optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.learning_rate)

max_grad_norm = 1.0

total_steps = len(training_loader) * Config.epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

for epoch in range(4):
    train(epoch)