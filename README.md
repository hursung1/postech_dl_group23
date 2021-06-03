# postech_dl_group23

210509 Update the training base code
        -Due to the limitations of the lab computer, I confirmed that it runs without errors, but I haven't finished training.

##In CNN_classificaition folder, there is code file named 'CNN_classification.ipynb', which is written by ipython notebook.
To run it, set data file of 'train_final.csv' and 'eval_final_open.csv' in the same folder.
After running, data file of 'sub_label.csv' that has infromation of label of test dataset.

##In RoBERTa_classification folder, there is code file named 'RoBERTa_classification.py'.
There are 4 models, model, model_a, model_b, model_c.

model is standard model for real label outputs {0, 1, 2, 3, 4}

model_a is adjusted model for positive{0, 1}, neutral{2}, negative{3, 4}

model_b is adjusted model for emphasized{0, 4}, somewhat{1, 3}, neutral{2}

model_c is model for simple MLP from input as model_a, model_b's outputs and output as real label {0, 1, 2, 3, 4}. It has layer as X -> 10 -> 5. X can change because we experiment not only with model_a and model_b.

Plot for just output of each real label and predicted label pair.

Change variable of Training, Training_A, Training_B, Training_C, Plot for training each of them.
You can choose desired epoch model from Load_Path_X variable.
