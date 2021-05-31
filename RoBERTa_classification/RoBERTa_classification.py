import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig
from transformers import get_linear_schedule_with_warmup

torch.cuda.empty_cache()

device_Gpu = False
if device_Gpu == True:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#####################
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_leaky_relu_stack = nn.Sequential(
            nn.Linear(8, 10),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Linear(10, 5),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )
    def forward(self, x):
        logits = self.linear_leaky_relu_stack(x).to(device)
        return logits
#####################

### parameters 
num_labels_A = 3
num_labels_B = 3
num_labels_C = 5
pad_size = 35
batch_size = 2
Valid_size = 1000
epochs = 10
seed_fix = False
Training = False
Training_A = False
Training_B = False
Training_C = False
Plot = True
loss_fn = nn.CrossEntropyLoss()
Test_epoch = 10
learning_rate = 2e-5


Model_version = "" #For fine tuning  ex) cased_v1, uncased_v2 ...
Save_version = ""  #v1, v2 ...
Pretrained_model ='roberta-base'       ## pretrained model

Load_Path_A = "./Trained_model_a/entire_model_test_uncased_epoch%d.pt" % 9
Load_Path_B = "./Trained_model_b/entire_model_test_uncased_epoch%d.pt" % 9
Load_Path_C = "./Trained_model_c/entire_model_test_uncased_epoch%d.pt" % 4

save_file_name = "./Results/submission_epoch%d.csv" % 9

save_file_name = save_file_name + Save_version

Reveiw_data =  pd.read_csv('./sentence-classification/train_final.csv')

sentences = Reveiw_data.Sentence.values
labels = Reveiw_data.Category.values

tokenizer = RobertaTokenizer.from_pretrained(Pretrained_model, do_lower_case=True)

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
train_data_labels, valid_data_labels= labels[Valid_size:], labels[:Valid_size]

# Train original label and save at Trained_model
if Training:
    train_dataset = TensorDataset(train_data_ids, train_data_masks, train_data_labels)
    valid_dataset = TensorDataset(valid_data_ids, valid_data_masks, valid_data_labels)

    train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_dataset),batch_size = batch_size)
    validation_dataloader = DataLoader(valid_dataset,sampler = SequentialSampler(valid_dataset),batch_size = batch_size)
    
    model = RobertaForSequenceClassification.from_pretrained(Pretrained_model,num_labels = num_labels_C, output_attentions = False, output_hidden_states = False)
    model.to(device)

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
                print('Epoch {}, Iteration {}, Loss {:.4f}'.format(epoch+1, step,total_train_loss/step))
    
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
        Save_Path = "./Trained_model_a/entire_model_test_uncased_epoch%d.pt" %(epoch+1)
        Save_Path = Save_Path +Model_version

        torch.save(model,Save_Path)
        print("Model saved")

# Train adjusted label which 0: 0,1(bad), 1: 3,4(good), 2: 2(so so) and save at Trained_model_a
if Training_A:
    for i in range(len(valid_data_labels)):
        if valid_data_labels[i] == 1:
            valid_data_labels[i] = 0 
        elif valid_data_labels[i] == 3 or valid_data_labels[i] == 4:
            valid_data_labels[i] = 1
    for i in range(len(train_data_labels)):
        if train_data_labels[i] == 1:
            train_data_labels[i] = 0
        elif train_data_labels[i] == 3 or valid_data_labels[i] == 4:
            train_data_labels[i] = 1

    train_dataset_a = TensorDataset(train_data_ids, train_data_masks, train_data_labels)
    valid_dataset_a = TensorDataset(valid_data_ids, valid_data_masks, valid_data_labels)

    train_dataloader_a = DataLoader(train_dataset_a,sampler = RandomSampler(train_dataset_a),batch_size = batch_size)
    validation_dataloader_a = DataLoader(valid_dataset_a,sampler = SequentialSampler(valid_dataset_a),batch_size = batch_size)
    
    model = RobertaForSequenceClassification.from_pretrained(Pretrained_model,num_labels = num_labels_A, output_attentions = False, output_hidden_states = False)
    model.to(device)

    optimizer = AdamW(model.parameters(),lr = learning_rate,eps = 1e-8)

    total_steps = len(train_dataloader_a) * epochs

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

        for step, batch in enumerate(train_dataloader_a):
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
                
                print('Epoch {}, Iteration {}, Loss {:.4f}'.format(epoch+1, step,total_train_loss/step))
    
        avg_train_loss = total_train_loss / len(train_dataloader_a)            
    
        
        print("training loss: {0:.4f}".format(avg_train_loss))
        
        print("###Validation###")
        model.eval()
        total_correct = 0
        total_label = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader_a:
        
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

        avg_val_loss = total_eval_loss / len(validation_dataloader_a)
        
       
        print("Validation Loss: {0:.4f}".format(avg_val_loss))
        Save_Path = "./Trained_model_a/entire_model_test_uncased_epoch%d.pt" %(epoch+1)
        Save_Path = Save_Path +Model_version

        torch.save(model,Save_Path)
        print("Model saved")

# Train adjusted label which 0: 0,4(very), 1: 1,3(normal), 2: 2(so so) and save at Trained_model_b
elif Training_B:
    for i in range(len(valid_data_labels)):
        if valid_data_labels[i] == 3:
            valid_data_labels[i] = 1 
        if valid_data_labels[i] == 4:
            valid_data_labels[i] = 0
    for i in range(len(train_data_labels)):
        if train_data_labels[i] == 3:
            train_data_labels[i] = 1
        if train_data_labels[i] == 4:
            train_data_labels[i] = 0

    train_dataset_b = TensorDataset(train_data_ids, train_data_masks, train_data_labels)
    valid_dataset_b = TensorDataset(valid_data_ids, valid_data_masks, valid_data_labels)

    train_dataloader_b = DataLoader(train_dataset_b,sampler = RandomSampler(train_dataset_b),batch_size = batch_size)
    validation_dataloader_b = DataLoader(valid_dataset_b,sampler = SequentialSampler(valid_dataset_b),batch_size = batch_size)
    
    model = RobertaForSequenceClassification.from_pretrained(Pretrained_model,num_labels = num_labels_B, output_attentions = False, output_hidden_states = False)
    model.to(device)

    optimizer = AdamW(model.parameters(),lr = learning_rate,eps = 1e-8)

    total_steps = len(train_dataloader_b) * epochs

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

        for step, batch in enumerate(train_dataloader_b):
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
    
        avg_train_loss = total_train_loss / len(train_dataloader_b)            
    
        
        print("training loss: {0:.4f}".format(avg_train_loss))
        
        print("###Validation###")
        model.eval()
        total_correct = 0
        total_label = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader_b:
        
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

        avg_val_loss = total_eval_loss / len(validation_dataloader_b)
        
       
        print("Validation Loss: {0:.4f}".format(avg_val_loss))
        Save_Path = "./Trained_model_b/entire_model_test_uncased_epoch%d.pt" %(epoch+1)
        Save_Path = Save_Path +Model_version

        torch.save(model,Save_Path)
        print("Model saved")

# Train by output of Trained_model_a, Trained_model_b and save at Trained_model_c
elif Training_C:
    print("## trained Model A load ##")
    model_A = torch.load(Load_Path_A)
    model_A.eval()
    prediction_labels_A = np.empty(0)

    print("## trained Model B load ##")
    model_B = torch.load(Load_Path_B)
    model_B.eval()
    prediction_labels_B = np.empty(0)

    model_C = NeuralNetwork().to(device)

    optimizer = AdamW(model_C.parameters(),lr = learning_rate,eps = 1e-8)

    train_dataset = TensorDataset(train_data_ids, train_data_masks, train_data_labels)
    valid_dataset = TensorDataset(valid_data_ids, valid_data_masks, valid_data_labels)

    train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_dataset),batch_size = batch_size)
    validation_dataloader = DataLoader(valid_dataset,sampler = SequentialSampler(valid_dataset),batch_size = batch_size)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    if seed_fix:
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        #torch.cuda.manual_seed_all(seed_val)

    for epoch in range(0, epochs):

        total_train_loss = 0

        print("###Training###")
        model_C.train()

        for step, batch in enumerate(train_dataloader):

            #  input ids, attention masks, labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():
                output_A = model_A(b_input_ids, attention_mask = b_input_mask)
                loss_A = output_A[0].cpu().numpy()
                output_B = model_B(b_input_ids, attention_mask = b_input_mask)
                loss_B = output_B[0].cpu().numpy()
                C_input_1 = np.reshape(np.append(loss_A[0], loss_B[0]),(1,8))
                C_input_2 = np.reshape(np.append(loss_A[1], loss_B[1]),(1,8))
                C_inputs = torch.Tensor(np.reshape(np.append(C_input_1, C_input_2),(2,8)))

            model_C.zero_grad()

            outputs = model_C(C_inputs)
            loss = loss_fn(outputs, b_labels)
        
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_C.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step % 200 == 0 and not step == 0:
                
                print('Epoch {}, Iteration {}, Loss {:.4f}'.format(epoch, step,total_train_loss/step))
    
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        print("training loss: {0:.4f}".format(avg_train_loss))
        
        print("###Validation###")
        model_C.eval()
        total_correct = 0
        total_label = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            total_label += len(b_labels)
            C_inputs = np.empty(0)

            with torch.no_grad():
                output_A = model_A(b_input_ids, attention_mask = b_input_mask)
                loss_A = output_A[0].cpu().numpy()
                output_B = model_B(b_input_ids, attention_mask = b_input_mask)
                loss_B = output_B[0].cpu().numpy()
                C_input_1 = np.reshape(np.append(loss_A[0], loss_B[0]),(1,8))
                C_input_2 = np.reshape(np.append(loss_A[1], loss_B[1]),(1,8))
                C_inputs = torch.Tensor(np.reshape(np.append(C_input_1, C_input_2),(2,8)))
                outputs = model_C(C_inputs)
                loss = loss_fn(outputs, b_labels)


            total_eval_loss += loss.item()
    
            prediction = torch.argmax(F.softmax(outputs), axis=1)

            correct = prediction.eq(b_labels)
        
            total_correct += correct.sum().item()

        avg_val_accuracy = total_correct / total_label
        
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
       
        print("Validation Loss: {0:.4f}".format(avg_val_loss))
        Save_Path = "./Trained_model_c/entire_model_test_uncased_epoch%d.pt" %(epoch+1)
        Save_Path = Save_Path +Model_version

        torch.save(model_C,Save_Path)
        print("Model saved")

# To see how miss.
elif Plot:
    train_dataset = TensorDataset(train_data_ids, train_data_masks, train_data_labels)
    valid_dataset = TensorDataset(valid_data_ids, valid_data_masks, valid_data_labels)

    train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_dataset),batch_size = batch_size)
    validation_dataloader = DataLoader(valid_dataset,sampler = SequentialSampler(valid_dataset),batch_size = batch_size)
    
    model = RobertaForSequenceClassification.from_pretrained(Pretrained_model,num_labels = num_labels_C, output_attentions = False, output_hidden_states = False)
    model.to(device)
    print("###Validation###")
    model = torch.load(Load_Path_A)
    model.eval()
    table = [0 for i in range(25)]
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
        for i in range(batch_size):
            table[prediction[i]*5+b_labels[i]] += 1

        correct = prediction.eq(b_labels)
        
        total_correct += correct.sum().item()
    for i in range(25):
        print(table[i])
    avg_val_accuracy = total_correct / total_label
        
    print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)   

#Load Trained_model_a,b,c and evaluate test cases.
else:
##################################################
# Kaggle Test data
    
    Reveiw_Test_data =  pd.read_csv('./sentence-classification/eval_final_open.csv')

    sentences = Reveiw_Test_data.Sentence.values
    id = Reveiw_Test_data.Id.values

    tokenizer = RobertaTokenizer.from_pretrained(Pretrained_model, do_lower_case=True)

    input_ids = []
    attention_masks = []

    for sent in sentences:

        encoded_dict = tokenizer.encode_plus(sent, add_special_tokens = True, max_length = pad_size,pad_to_max_length = True,return_attention_mask = True,return_tensors = 'pt' )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    test_dataset = TensorDataset(input_ids, attention_masks)
    test_dataloader = DataLoader(test_dataset,sampler = SequentialSampler(test_dataset),batch_size = 1)

    print("## trained Model load ##")
    model_A = torch.load(Load_Path_A)
    model_B = torch.load(Load_Path_B)
    model_C = torch.load(Load_Path_C)

    model_A.eval()
    model_B.eval()
    model_C.eval()

    prediction_labels = np.empty(0)
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device) 
        
        with torch.no_grad():        
            output_A = model_A(b_input_ids, attention_mask = b_input_mask)
            loss_A = output_A[0].cpu().numpy()
            output_B = model_B(b_input_ids, attention_mask=b_input_mask)
            loss_B = output_B[0].cpu().numpy()
            C_input = torch.Tensor(np.reshape(np.append(loss_A, loss_B),(1, 8)))
            outputs = model_C(C_input)

        prediction = torch.argmax(F.softmax(outputs), axis=1)
        prediction = prediction.to('cpu').numpy()
        test_id = b_input_ids.to('cpu').numpy()
        prediction_labels = np.concatenate((prediction_labels,prediction))
        
    prediction_labels = np.int64(prediction_labels)
    

    ## Export data to csv
    test_ids = list(range(0,len(prediction_labels)))
    data_export = {'Id':test_ids,'Category':prediction_labels}

    write_csv = pd.DataFrame(data_export)

    write_csv.to_csv(save_file_name,index=False)


