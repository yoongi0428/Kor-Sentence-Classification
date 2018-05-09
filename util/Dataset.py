import os
import numpy as np
from util.kor_parser import decompose_str_as_one_hot
from util.kor_eum_parser import decompose_str_as_one_hot_eum

class Dataset(object):

    def __init__(self, dataset_path, num_classes=2, eumjeol=False, max_len=420):
        postfix = {2: 'binary', 5: 'five'}
        self.PATH = dataset_path
        self.data_file = 'data_' + postfix[num_classes]
        self.label_file = 'label_' + postfix[num_classes]
        self.eumjeol = eumjeol
        self.strmaxlen = max_len

        self.read_data()

    def read_data(self):
        with open(os.path.join(self.PATH, self.data_file), 'rt', encoding='utf-8') as f:
            reviews = preprocess(f.readlines(), self.strmaxlen, self.eumjeol)
        with open(os.path.join(self.PATH,self.label_file), 'rt', encoding='utf-8') as f:
            ratings = [np.float32(x) for x in f.readlines()]
        ratings = np.array(ratings)

        p = np.arange(len(reviews))
        np.random.shuffle(p)
        reviews = reviews[p, :]
        ratings = ratings[p]

        train_num = int(len(reviews) * 0.8)
        train_data = reviews[:train_num, :]
        train_label = ratings[:train_num]
        test_data = reviews[train_num:, :]
        test_label = ratings[train_num:]
        self.train = train_data, train_label
        self.test = test_data, test_label

    def __len__(self):
        return len(self.train[0])

    def __getitem__(self, idx):
        return self.train[0][idx], self.train[1][idx]

    def shuffle(self):
        p = np.arange(len(self.train[0]))
        np.random.shuffle(p)
        data = self.train[0]
        label = self.train[1]
        self.train = data[p, :], label[p]

def preprocess(data, max_len, eumjeol):
    vectorized_data = [decompose_str_as_one_hot(s) if not eumjeol else decompose_str_as_one_hot_eum(s) for s in data]
    padded_data = np.zeros((len(data), max_len), dtype=np.int32)
    for i, s in enumerate(vectorized_data):
        length = len(s)
        if length < max_len:
            padded_data[i, :length] = np.array(s)
        else:
            padded_data[i:, :max_len] = np.array(s)[:max_len]   # Truncate from the front
            # padded_data[i, :max_len] = np.array(s)[-max_len:]   # Truncate from the back
    return padded_data