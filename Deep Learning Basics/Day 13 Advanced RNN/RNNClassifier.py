import csv
import gzip
import math

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import time

BATCH_SIZE = 256
HIDDEN_SIZE = 100
N_LAYER = 2
N_EPOCHS = 20
N_CHARS = 128  # ASCII
N_COUNTRY = 18
USE_GPU = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device('cuda')
        tensor = tensor.to(device)
    return tensor


def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class NameDataSet(Dataset):
    def __init__(self, is_train_set=True):
        filename = '../resources/names_train.csv.gz' if is_train_set else '../resources/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dic = self.getCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, item):
        return self.names[item], self.country_dic[self.countries[item]]

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dic = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dic[country_name] = idx
        return country_dic

    def idx2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirection=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirection else 1
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirection)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros((self.n_layers * self.n_directions, batch_size, self.hidden_size))
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        # input shape : B x S -> S x B
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        # pack them up
        # gru_input = pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(embedding, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output


def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()

    #  make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    #  sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


trainset = NameDataSet(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataSet(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
B_COUNTRY = trainset.getCountriesNum()

classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
classifier.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=classifier.parameters(), lr=0.001)


def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 20 == 0:
            print('Epoch: ', epoch)
            print('Loss: ', total_loss)

    return total_loss


def name_test_sets_model():
    correct = 0
    total = len(testset)
    print('Evaluating trained model ... ')
    print('Please stand by ... ')
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print('Test Set Accuracy: ', percent, '%')
    return correct / total


if __name__ == '__main__':
    start = time.time()
    print('Total Epoch: ', N_EPOCHS)
    acc_list = []
    epoch_list = []
    for epoch in range(1, N_EPOCHS + 1):
        trainModel()
        acc = name_test_sets_model()
        acc_list.append(acc)
        epoch_list.append(epoch)
    plt.plot(epoch_list, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on Test Sets')
    plt.grid()
    plt.show()

