
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F

torch.cuda.empty_cache()

class ReviewDataset(Dataset):

    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)
    def __getitem__(self,ind):
        Label = self.data.iloc[ind,1]
        Sentence = self.data.iloc[ind,2]
        return Label,Sentence


### Initial code of BERT
## Load data set
train_data = pd.read_csv('./sentence-classification/train_final.csv')
train_data.dropna(inplace=True)
#train_data = train_data.sample(frac = 0.3)


Rev_train = ReviewDataset(train_data)
train_loader = DataLoader(Rev_train,batch_size = 7, shuffle = True)

## Configuring the BERT model

#device = torch.device("cuda")
total_label = 5
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels= total_label)

#model.to(device)
optimizer = Adam(model.parameters(),lr=1e-6)

iter = 1
check_iter = 500
epochs = 1
total_loss = 0
total_len = 0
total_correct = 0

model.train()

for epoch in range(epochs):

    for Label,Sentence in train_loader:
        optimizer.zero_grad()

        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in Sentence]
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
    
        sample = torch.LongTensor(padded_list)
        
        Labels = torch.LongTensor(Label)
        
        outputs = model(sample, labels=Labels)
        loss = outputs[0]
        logits = outputs[1]
      
        prediction = torch.argmax(F.softmax(logits), dim=1)
        #print(logits, Labels, prediction)
        #break
        correct = prediction.eq(Labels)

        total_correct += correct.sum().item()
        total_len += len(Labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    if iter % check_iter == 0:
            print('Iteration {}_Training Loss: {:.4f}, Accuracy: {:.4f}'.format(iter, total_loss/check_iter, total_correct/total_len))
            total_loss = 0
            total_len = 0
            total_correct = 0
    iter = iter +1