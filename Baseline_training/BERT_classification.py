import pandas as pd
import numpy as np
import random

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

torch.cuda.empty_cache()

device_Gpu = True
if device_Gpu == True:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

### parameters 
num_labels = 5
pad_size = 35
batch_size = 2
Valid_size = 1000
epochs = 10
seed_fix = True
Training =  False

Test_epoch = 10
learning_rate = 2e-5


Model_version = "" #For fine tuning  ex) cased_v1, uncased_v2 ...
Save_version = ""  #v1, v2 ...
Pretrained_model ='bert-base-uncased'       ## pretrained model
Load_Path = "./Trained_model/entier_model_test_uncased_epoch%d.pt" % Test_epoch

save_file_name = "./Results/submission_epoch%d.csv" % Test_epoch

save_file_name = save_file_name+Save_version
Load_Path = Load_Path + Model_version

Reveiw_data =  pd.read_csv('./sentence-classification/train_final.csv')

sentences = Reveiw_data.Sentence.values
labels = Reveiw_data.Category.values

tokenizer = BertTokenizer.from_pretrained(Pretrained_model, do_lower_case=True)

input_ids = []
attention_masks = []

#tokenization of the sentence
for sent in sentences:

    encoded_dict = tokenizer.encode_plus(sent, add_special_tokens = True, max_length = pad_size,pad_to_max_length = True,return_attention_mask = True,return_tensors = 'pt' )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)


train_data_ids, valid_data_ids= input_ids[Valid_size:], input_ids[:Valid_size]
train_data_masks, valid_data_masks= attention_masks[Valid_size:], attention_masks[:Valid_size]
train_data_lables, valid_data_labels= labels[Valid_size:], labels[:Valid_size]

train_dataset = TensorDataset(train_data_ids, train_data_masks, train_data_lables)
valid_dataset =  TensorDataset(valid_data_ids, valid_data_masks, valid_data_labels)



train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_dataset),batch_size = batch_size)
validation_dataloader = DataLoader(valid_dataset,sampler = SequentialSampler(valid_dataset),batch_size = batch_size)
if Training:

    model = BertForSequenceClassification.from_pretrained(Pretrained_model,num_labels = num_labels,output_attentions = False,output_hidden_states = False)
    model.cuda()    

    optimizer = AdamW(model.parameters(),lr = learning_rate,eps = 1e-8)

    total_steps = len(train_dataloader) * epochs


    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    if seed_fix:
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

    for epoch in range(0, epochs):
        

        total_train_loss = 0

        print("###Training###")
        model.train()

        for step, batch in enumerate(train_dataloader):

            #  input ids, attention masks, labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
        
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step % 200 == 0 and not step == 0:
                
                print('Epoch {}, Iteration {}, Loss {:.4f}'.format(epoch, step,total_train_loss/step))
    
        avg_train_loss = total_train_loss / len(train_dataloader)            
    
        
        print("training loss: {0:.4f}".format(avg_train_loss))
        
        print("###Validation###")
        model.eval()
        total_correct = 0
        total_label = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
        
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            total_label += len(b_labels)
            with torch.no_grad():        

                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                logits = outputs[1]
            total_eval_loss += loss.item()
    
            prediction = torch.argmax(F.softmax(logits), axis=1)


            correct = prediction.eq(b_labels)
        
            total_correct += correct.sum().item()

        avg_val_accuracy = total_correct / total_label
        
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
       
        print("Validation Loss: {0:.4f}".format(avg_val_loss))
        Save_Path = "./Trained_model/entier_model_test_uncased_epoch%d.pt" %(epoch+1)
        Save_Path = Save_Path +Model_version

        torch.save(model,Save_Path)
        print("Model saved")
    
else:
##################################################
# Kaggle Test data
    
    Reveiw_Test_data =  pd.read_csv('./sentence-classification/eval_final_open.csv')

    sentences = Reveiw_Test_data.Sentence.values
    id = Reveiw_Test_data.Id.values

    tokenizer = BertTokenizer.from_pretrained(Pretrained_model, do_lower_case=True)

    input_ids = []
    attention_masks = []

    for sent in sentences:

        encoded_dict = tokenizer.encode_plus(sent, add_special_tokens = True, max_length = pad_size,pad_to_max_length = True,return_attention_mask = True,return_tensors = 'pt' )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    test_dataset = TensorDataset(input_ids, attention_masks)
    test_dataloader = DataLoader(test_dataset,sampler = SequentialSampler(test_dataset),batch_size = batch_size)

    print("## trained Model load ##")
    model = torch.load(Load_Path)

    model.eval()
    prediction_labels = np.empty(0)
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        
        
        with torch.no_grad():        

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
        

        prediction = torch.argmax(F.softmax(logits), axis=1)
        prediction = prediction.to('cpu').numpy()
        test_id = b_input_ids.to('cpu').numpy()
        prediction_labels = np.concatenate((prediction_labels,prediction))
        
    prediction_labels = np.int64(prediction_labels)
    

    ## Export data to csv
    test_ids = list(range(0,len(prediction_labels)))
    data_export = {'Id':test_ids,'Category':prediction_labels}

    write_csv = pd.DataFrame(data_export)

    write_csv.to_csv(save_file_name,index=False)