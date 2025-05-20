import torch 
from torch.utils.data import Dataset
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import os
import numpy as np
import torch.nn as nn
import random

# Special tokens
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"
UNK_TOKEN = "<UNK>"

SEED = 42  # or any fixed number you like
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Load the lexicon data
types = {"train":'hi.translit.sampled.train.tsv',"val":'hi.translit.sampled.dev.tsv',"test":"hi.translit.sampled.test.tsv"}
with open(os.path.join("lexicons/",types["train"]), "r", encoding="utf-8") as f:
    lines = f.readlines()
train_data = np.array([[text.split("\t")[0],text.split("\t")[1][:-1]] for text in lines if not text.split("\t")[0] == '</s>'])
with open(os.path.join("lexicons/",types["val"]), "r", encoding="utf-8") as f:
    lines = f.readlines()
val_data = np.array([[text.split("\t")[0],text.split("\t")[1][:-1]] for text in lines if not text.split("\t")[0] == '</s>'])
with open(os.path.join("lexicons/",types["test"]), "r", encoding="utf-8") as f:
    lines = f.readlines()
test_data = np.array([[text.split("\t")[0],text.split("\t")[1][:-1]] for text in lines if not text.split("\t")[0] == '</s>'])
test_data_point = np.array([["अनुज","anuj"],["निर्णयप्रक्रियेत","nirnayaprakriyet"]])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
merged_data = np.concatenate((train_data,val_data))


class LangDataset(Dataset):
    """
    Dataset for Latin and Devanagari words."""
    def __init__(self,type:str):
        types = {"train":train_data,"val":val_data,"test":test_data, "test_ponit":test_data_point}
        data = types[type]
        self.X,self.Y,self.X_encoded,self.Y_encoded = [],[],[],[]
        for word in data:
            self.X.append(word[1])
            self.Y.append(word[0])
            self.X_encoded.append(tokenise(word[1],latinList2int))
            self.Y_encoded.append(tokenise(word[0],devnagri2int))
        
    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        latin_word= self.X[idx]
        devnagri_word = self.Y[idx]
        latin_tensor = self.X_encoded[idx]
        devnagri_tensor = self.Y_encoded[idx]

        return latin_word, devnagri_word, latin_tensor, devnagri_tensor

    def __len__(self):
        return len(self.X)
    
# Update the function to create mappings to include the special tokens
def create_mappings(vocab):
    """
    Create mappings from words to integers and vice versa."""
    vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(vocab)
    word2int = {word: i for i, word in enumerate(vocab)}
    int2word = {i: word for word, i in word2int.items()}
    return word2int, int2word

def wordEncoder(words,encodelist):
    """
    Encode words into a tensor using one-hot encoding."""
    n_letters = len(encodelist)
    tensor = torch.zeros(len(words), n_letters)
    for i,word in enumerate(words):
        tensor[i][encodelist[word]] = 1
    return tensor
    
def tokenise(word, wordMap):
    """
    Tokenize a word into a tensor using the provided word map."""
    return torch.tensor([wordMap[SOS_TOKEN]] + [wordMap[letter] for letter in word] + [wordMap[EOS_TOKEN]], dtype=torch.long)

def asMinutes(s):
    """Convert seconds to minutes and seconds."""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    """Return a string representing the elapsed time since a given time."""
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    """Plot the given points."""
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def collate_fn(batch):
    """
    Custom collate function to handle padding and batching of sequences."""
    # Sort by input sequence length (descending)
    batch.sort(key=lambda x: len(x[2]), reverse=True)
    
    latin_words, devnagri_words, latin_tensors, devnagri_tensors = zip(*batch)
    
    # Get sequence lengths
    input_lengths = [len(seq) for seq in latin_tensors]
    target_lengths = [len(seq) for seq in devnagri_tensors]
    
    # Pad sequences
    latin_tensors = nn.utils.rnn.pad_sequence(latin_tensors, batch_first=True, padding_value=latinList2int[PAD_TOKEN])
    devnagri_tensors = nn.utils.rnn.pad_sequence(devnagri_tensors, batch_first=True, padding_value=devnagri2int[PAD_TOKEN])
    
    return (latin_words, devnagri_words, 
            latin_tensors.to(device), devnagri_tensors.to(device),
            input_lengths, target_lengths)

devnagri2int, int2devnagri = create_mappings(set("".join(merged_data[:, 0])))
latinList2int, int2latinList = create_mappings(set("".join(merged_data[:, 1])))