import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, FastText, CharNGram

import pandas as pd
import os

import train
from data import SentimentDataset 
from model import SentimentClassificationModel

EPOCH = 10
batch_size = 64
emb_dim = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(params):
    # build dataset
    train_data = pd.read_csv('./data/train_final.csv')
    tokenizer = get_tokenizer('spacy', language='en')

    # embedding = CharNGram()
    # embedding = FastText()
    # embedding = GloVe() # use glove embedding with default option(name='840B', dim=300)
    embedding = GloVe(name='6B', dim=str(emb_dim))

    train_data, val_data = train_data[500:], train_data[:500]
    train_dataset = SentimentDataset(train_data, tokenizer, embedding)
    val_dataset = SentimentDataset(val_data, tokenizer, embedding)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = SentimentClassificationModel(emb_dim, 256, 0.3).to(device)
    crit = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=5e-3)

    best_val_acc = 0
    early_stop_cnt = 0
    epoch = 0
    while early_stop_cnt != 5:
        train.trainer(epoch, model, train_dataloader, crit, optim, device)
        val_acc = train.eval(epoch, model, val_dataloader, device)
        if val_acc > best_val_acc and epoch > 0:
            torch.save(model.state_dict(), './model/lstm_best.pt')
            best_val_acc = val_acc
            early_stop_cnt = 0
        
        early_stop_cnt += 1
        epoch += 1

    print("Early stopping condition satisfied")

def test(params):
    tokenizer = get_tokenizer('spacy', language='en')
    embedding = GloVe(name='6B', dim=str(emb_dim))

    test_data = pd.read_csv('./data/eval_final_open.csv')
    test_dataset = SentimentDataset(test_data, tokenizer, embedding, True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SentimentClassificationModel(emb_dim, 256, 0.3).to(device)
    model.load_state_dict(torch.load('./model/lstm_best.pt'))

    train.eval(0, model, test_dataloader, device)


if __name__ == "__main__":
    if not os.path.isdir("./data"):
        os.mkdir("./data")

    if True:
        main(None)

    else:
        test(None)