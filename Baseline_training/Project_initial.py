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
## parameter 
Batch_size = 7
Valid_size = 1000

Pad_size = 35

## Load data set
train_data = pd.read_csv('./sentence-classification/train_final.csv')
train_data, val_data = train_data[Valid_size:], train_data[:Valid_size]



Rev_train = ReviewDataset(train_data)
Rev_valid = ReviewDataset(val_data)
train_loader = DataLoader(Rev_train,batch_size = Batch_size, shuffle = True)
valid_loader = DataLoader(Rev_valid,batch_size = Batch_size, shuffle = False)



## Configuring the BERT model

#device = torch.device("cuda")
total_label = 5                                                                                          # In csv file, there are 5 categories
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')                                             # Various pretrained BERT model, https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
model = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels= total_label)         # For multi-classification

#model.to(device)
optimizer = Adam(model.parameters(),lr=1e-6)

iter = 1
check_iter = 10
epochs = 1
total_loss = 0
total_len = 0
total_correct = 0

model.train()

for epoch in range(epochs):

    for Label,Sentence in train_loader:
        optimizer.zero_grad()
            ## Bert Tokenizer encoding => [CLS] , Token1, Token2,... ,Token N, [SEP] => if max_length = 35, then N=33
        encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length = Pad_size) for t in Sentence]
        
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

# evaluation

model.eval()

total_loss = 0
total_len = 0
total_correct = 0
for Label,Sentence in valid_loader:
    encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length = Pad_size) for t in Sentence]
    padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
    sample = torch.LongTensor(padded_list)
    Labels = torch.LongTensor(Label)    
    outputs = model(sample, labels=Labels)

    logits = outputs[1]
    prediction = torch.argmax(F.softmax(logits), dim=1)
    correct = prediction.eq(Labels)
    total_correct += correct.sum().item()
    total_len += len(Labels)

print('Validation Accuracy :', total_correct / total_len)