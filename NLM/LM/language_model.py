import math 
import torch
import argparse
from utils import tokenize
from model import Model
from dataset import Dataset
import sys
sys.argv=['']
del sys

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_probability(dataset, model, text):
    model.eval()

    text = tokenize(text)
    words = text.split(' ')
    for i in range(len(words)):
      if words[i] not in dataset.get_uniq_words():
        words[i] = "<unk>"
    prob = 1
    for i in range(1, len(words)):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[:i]]]).to(device)

        state_h, state_c = model.init_state(i)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        print(last_word_logits)
        p = torch.nn.functional.softmax(last_word_logits, dim=0).cpu().detach().numpy()
        word_index = dataset.word_to_index[words[i]]
        # print(dataset.index_to_word[word_index], p[word_index])
        prob *= p[word_index]

    return prob

def get_perplexity(dataset, model, text):
    return math.pow(1/get_probability(dataset, model, text), 1/len(text.split()))

def handle_unknowns(words):
          word_counts = Counter(words)
          for i in range(len(words)):
            if word_counts[words[i]] <= 2:
              words[i] = "<unk>"
          return words

def perp_to_file():
  File_object = open(file_path + "/europarl-corpus/test.europarl" ,"r")
  data = File_object.readlines()

  sum = 0
  perps = []
  count = 0
  for line in data:
    try:
      x = get_perplexity(dataset, model, line)
      sum += x
      count += 1
      print(count)
      perps.append(line.rstrip("\n") + "\t" + str(x) + "\n")
      print(line.rstrip("\n") + "\t" + str(x) + "\n")
    except: 
      print("error, moving to next line")
  with open( "LM_output2.txt" , 'w', encoding="utf8") as f:
    f.write("Average perplexity: " + str(sum/1000)  + "\n")
    f.writelines(perps)

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
parser.add_argument('--dataset', type=str, default="train.europarl")
args = parser.parse_args()

dataset = Dataset(args)
model = Model(dataset)
model.to(device)

input_path = sys.argv[1]

model = torch.load(input_path)

sentence = input("Input Sentence: ")
print(get_probability(dataset, model, sentence))



