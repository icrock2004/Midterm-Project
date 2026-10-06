import numpy as np
import pandas as pd
import spacy
import ast
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import copy
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PATH = "dataset.txt"

def fix_surrogates(text):
    # fixes errors caused by emojis
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")

rows = []
with open(PATH, encoding="utf-8", newline="") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, inp, out = ast.literal_eval(line)   # safe parser, handles \n and \u escapes
        rows.append({"llm_name": name, "llm_input": fix_surrogates(inp), "llm_output": fix_surrogates(out)})

df = pd.DataFrame(rows)

# debugging
#print(df.shape)                      # (2757, 3)
#print(df["llm_name"].value_counts()) # 919 each
#print(df.isna().sum())               # makes sure no missing values
df = df.dropna(subset=["llm_name", "llm_output"]).reset_index(drop=True)

# Encode labels and split
le = LabelEncoder()
df["label"] = le.fit_transform(df["llm_name"])

prompts = df["llm_input"].unique()

train_p, temp_p = train_test_split(prompts, test_size=0.3, random_state=42)
val_p, test_p   = train_test_split(temp_p,  test_size=0.5, random_state=42)

train_df = df[df["llm_input"].isin(train_p)]
val_df   = df[df["llm_input"].isin(val_p)]
test_df  = df[df["llm_input"].isin(test_p)]

# tokenize
nlp = spacy.blank("en")  # tokenizer only, fast

def tokenize(texts):
    out = []
    for doc in nlp.pipe(texts, batch_size=256):
        toks = []
        for t in doc:
            if t.is_space:
                if "\n" in t.text:
                    toks.append("<NL>")   # keep paragraph/list structure
            else:
                toks.append(t.text)       # keep case, punctuation
        out.append(toks)
    return out

train_tokens = tokenize(train_df["llm_output"])
val_tokens   = tokenize(val_df["llm_output"])
test_tokens  = tokenize(test_df["llm_output"])

# Build vocabulary from training data
# counts how often each token appears in training
counts = Counter()
for tokens in train_tokens:
    counts.update(tokens)

# 0 = padding and 1 = unknown word, then adds every token that's seen at least twice
vocab = {"<pad>": 0, "<unk>": 1}
for token, count in counts.items():
    if count >= 2:
        vocab[token] = len(vocab)

#print("Vocab size:", len(vocab))

max_len = int(np.percentile([len(t) for t in train_tokens], 95))
#print("max_len:", max_len)

def to_ids(token_lists):
    all_ids = []
    all_lengths = []
    for tokens in token_lists:
        # unknown words become 1
        ids = []
        for tok in tokens:
            ids.append(vocab.get(tok, 1))
        ids = ids[:max_len]                           # cut if too long
        all_lengths.append(max(len(ids), 1))          # real length, before padding
        ids = ids + [0] * (max_len - len(ids))        # pad with 0s if too short
        all_ids.append(ids)
    return torch.tensor(all_ids), torch.tensor(all_lengths)

X_train, len_train = to_ids(train_tokens)
X_val, len_val = to_ids(val_tokens)
X_test, len_test = to_ids(test_tokens)

y_train = torch.tensor(train_df["label"].values)
y_val = torch.tensor(val_df["label"].values)
y_test = torch.tensor(test_df["label"].values)

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim = 100, num_filters=100, kernel_sizes=(2,3,4,5), num_classes=3, dropout=0.5):
        super().__init__()

        # embed token IDs to word vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.2)

        # one conv block per kernel size
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            block = nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, kernel_size=k),
                nn.BatchNorm1d(num_filters),
                nn.ReLU()
            )
            self.convs.append(block)

        # final linear layer, one score per llm
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x, lengths=None):          
        e = self.embedding(x)                    
        e = self.embed_dropout(e)
        e = e.transpose(1, 2)                   

        pooled = []

        # max-pool: keep the strongest match per filter
        for conv in self.convs:
            features = conv(e)                  
            strongest = features.max(dim=2).values   
            pooled.append(strongest)

        out = torch.cat(pooled, dim=1)          
        out = self.dropout(out)
        return self.fc(out)                      


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

BATCH_SIZE = 32

# each batch gives (token IDs, lengths, labels) together
train_loader = DataLoader(TensorDataset(X_train, len_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, len_val, y_val), batch_size=BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(X_test, len_test, y_test), batch_size=BATCH_SIZE)

loss_fn = nn.CrossEntropyLoss()

# pass one time over training data and update weights
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, len_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad() # clear old gradients
        outputs = model(X_batch, len_batch)
        loss = loss_fn(outputs, y_batch) # find loss
        loss.backward() # compute new gradients
        optimizer.step() # update weights

        total_loss += loss.item() * len(y_batch)
        correct += (outputs.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total
