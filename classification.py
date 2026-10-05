import numpy as np
import pandas as pd
import spacy
import ast
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

# sanity checks
#print(df.shape)                      # (2757, 3)
#print(df["llm_name"].value_counts()) # 919 each
#print(df.isna().sum())               # makes sure no missing values
df = df.dropna(subset=["llm_name", "llm_output"]).reset_index(drop=True)

# Encode labels and split
le = LabelEncoder()
df["label"] = le.fit_transform(df["llm_name"])

prompts = df["llm_input"].unique()

train_p, temp_p = train_test_split(prompts, test_size=0.2, random_state=42)
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
print("max_len:", max_len)

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

# Embedding layer
embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=128, padding_idx=0)