from torch.utils.data import Dataset

PAD_SIZE = 35

class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, embedding, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.is_train = is_train
        # True  --> dataset consists of data and label
        # False --> dataset consists of data only

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.tokenizer(self.data.iloc[index]['Sentence'])
        
        # pad sentence data
        for _ in range(len(x), PAD_SIZE):
            x.append('<pad>')
        
        # trim sentence data
        x = x[:PAD_SIZE] 
        x = self.embedding.get_vecs_by_tokens(x)

        if self.is_train:
            y = self.data.iloc[index]['Category']
            return x, y
        else: 
            return x