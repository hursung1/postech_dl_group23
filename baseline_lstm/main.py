import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, FastText, CharNGram

import matplotlib.pyplot as plt
import pandas as pd
import os

import train
from data import SentimentDataset 
from model import SentimentClassificationModel
from config import get_params

EPOCH = 10
batch_size = 64
emb_dim = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plotting(title, xlabel, ylabel, data, dpi=500):
    plt.figure()
    plt.title(title)
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{title}.png", dpi=dpi)
    plt.close()


def main(params):
    # build dataset
    train_data = pd.read_csv('./data/train_final.csv')
    tokenizer = get_tokenizer('spacy', language='en')

    if params.emb_type == "GloVe":
        embedding = GloVe(name=params.emb_data, dim=params.emb_dim) # use glove embedding with default option(name='840B', dim=300)
    elif params.emb_type == "CharNGram":
        embedding = CharNGram()
    elif params.emb_type == "FastText":
        embedding = FastText(name=params.emb_data, dim=params.emb_dim)
    else:
        print("Wrong embedding type")
        exit()

    train_data, val_data = train_data[1000:], train_data[:1000]
    train_dataset = SentimentDataset(train_data, tokenizer, embedding)
    val_dataset = SentimentDataset(val_data, tokenizer, embedding)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = SentimentClassificationModel(params.emb_dim, params.hidden_dim, params.dropout).to(device)
    crit = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    best_val_acc = 0
    early_stop_cnt = 0
    epoch = 0
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    while early_stop_cnt != 5:
        loss_list, train_acc = train.trainer(epoch, model, train_dataloader, crit, optim, device)
        val_acc = train.eval(epoch, model, val_dataloader, device, False)
        if val_acc > best_val_acc and epoch > 0:
            torch.save(model.state_dict(), './model/lstm_best.pt')
            best_val_acc = val_acc
            early_stop_cnt = 0
        
        early_stop_cnt += 1
        epoch += 1
        train_loss_list.extend(loss_list)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    print("Early stopping condition satisfied")
    plotting("train_loss", "steps", "loss", train_loss_list)
    plotting("train_accuracy", "epoch", "accuracy", train_acc_list)
    plotting('validation_accuracy', "epoch", "accuracy", val_acc_list)

def test(params):
    tokenizer = get_tokenizer('spacy', language='en')
    embedding = GloVe(name=params.emb_data, dim=params.emb_dim)

    test_data = pd.read_csv('./data/eval_final_open.csv')
    test_dataset = SentimentDataset(test_data, tokenizer, embedding, False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SentimentClassificationModel(params.emb_dim, params.hidden_dim, 0.3).to(device)
    model.load_state_dict(torch.load('./model/lstm_best.pt'))

    inference = {'Id': [i for i in range(len(test_data))]}
    inference['Category'] = train.eval(0, model, test_dataloader, device, True)

    df = pd.DataFrame(inference)
    df.to_csv("./data/out.csv", index=False)


if __name__ == "__main__":
    params = get_params()

    if not os.path.isdir("./data"):
        os.mkdir("./data")

    params.is_train = bool(params.is_train)
    if params.is_train:
        main(params)

    else:
        test(params)