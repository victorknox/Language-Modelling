import spacy
import torch
import pandas as pd
from collections import Counter
from utils import tokenize


file_path = 'drive/MyDrive/intro-to-nlp-assign3'
device = "cuda" if torch.cuda.is_available() else "cpu"

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        def handle_unknowns(words):
          word_counts = Counter(words)
          for i in range(len(words)):
            if word_counts[words[i]] <= 2:
              words[i] = "<unk>"
          return words
        File_object = open(file_path + "/europarl-corpus/" + args.dataset,"r")
        data = File_object.readlines()
        text = ""
        # adding end of sequence to make the model learn seperation between sentences
        for line in data:
          text += "<sos> " + tokenize(line) + " <eos> "
        return handle_unknowns(text.split(' '))

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (torch.tensor(self.words_indexes[index:index+self.args.sequence_length]).to(device), torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]).to(device))