import numpy as np
import pandas as pd
import spacy
import ast
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import copy
import matplotlib.pyplot as plt



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
df = df.drop_duplicates(subset=["llm_input", "llm_output"]).reset_index(drop=True)

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

# 0 = padding and 1 = unknown word, then adds every token that's seen at least five times
vocab = {"<pad>": 0, "<unk>": 1}
for token, count in counts.items():
    if count >= 5:
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

# class for CNN classifier
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

# class for the LSTM classifier
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=1, num_classes=3, dropout=0.5, embed_dropout=0.2):
        super().__init__()

        # embed token IDs to word vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(embed_dropout)

        # bidirectional LSTM reads the response forward and backward to make sure it gets common openers and closers
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,   # only applies between stacked layers
        )

        # final linear layer, one score per llm
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)   # times two for forward and backward

    def forward(self, x, lengths):
        e = self.embedding(x)                        
        e = self.embed_dropout(e)

        # pack so the LSTM skips the padding and only reads each real response
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (hidden_state, _) = self.lstm(packed)

        # final hidden state of the last layer, forward and backward
        h = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)     # (batch, hidden_dim * 2)
        h = self.dropout(h)
        return self.fc(h) 
    

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


# evaluate on either the validation or test data without updating weights
def evaluate(model, loader):
    model.eval()                                   # turns dropout off
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():                          # don't need gradients 
        for X_batch, len_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch, len_batch)
            loss = loss_fn(outputs, y_batch)

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * len(y_batch)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels

# the actual training run, keeps the best model found
def train_model(model, epochs=15, lr=1e-3, weight_decay=0.0, use_scheduler=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f} | lr {current_lr:.1e}")

        if scheduler:
            scheduler.step(val_loss)

        # save the weights from the epoch with the lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)            # go back to the best epoch
    return history

# loss and accuracy graphs
def plot_history(history, title, filename):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title(f"{title} - loss"); ax1.set_xlabel("epoch"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_title(f"{title} - accuracy"); ax2.set_xlabel("epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# saves the finetuning results to a csv to see which parameters were most successful
def save_results(results, best, name):
    table = pd.DataFrame(results).sort_values("val_loss")
    print(table.to_string(index=False))
    table.to_csv(f"{name}_tuning.csv", index=False)
    pd.DataFrame(best["history"]).to_csv(f"{name}_history.csv", index_label="epoch")
    return best["model"], best["history"]

def run_one(model, lr, epochs, settings, results, best,
            weight_decay=0.0, use_scheduler=False):
    history = train_model(model, epochs=epochs, lr=lr,
                          weight_decay=weight_decay, use_scheduler=use_scheduler)

    best_epoch = int(np.argmin(history["val_loss"]))
    val_loss = history["val_loss"][best_epoch]

    results.append({
        **settings, "lr": lr,
        "weight_decay": weight_decay, "scheduler": use_scheduler,
        "best_epoch": best_epoch + 1,
        "val_loss": round(val_loss, 4),
        "val_acc": round(history["val_acc"][best_epoch], 3),
    })
    # model already holds its best-epoch weights, so we can keep it directly
    if val_loss < best["val_loss"]:
        best.update(val_loss=val_loss, model=model, history=history)

def tune_cnn(epochs=20):
    results, best = [], {"val_loss": float("inf")}

    for dropout in [0.3, 0.5, 0.6]:
        for kernels in [(2, 3, 4), (2, 3, 4, 5)]:
            for sched in [False, True]:
                print(f"\n--- CNN: dropout={dropout}, kernels={kernels}, scheduler={sched} ---")
                torch.manual_seed(42)
                model = CNNClassifier(vocab_size=len(vocab), dropout=dropout,
                                      kernel_sizes=kernels).to(device)
                settings = {"dropout": dropout, "kernels": str(kernels)}
                run_one(model, 1e-3, epochs, settings, results, best,
                        use_scheduler=sched)

    return save_results(results, best, "cnn")


def tune_lstm(epochs=15):
    results, best = [], {"val_loss": float("inf")}

    for dropout in [0.5, 0.6]:
        for hidden in [64, 128]:
            for wd in [0.0, 1e-3, 1e-4]:
                print(f"\n--- LSTM: dropout={dropout}, hidden_dim={hidden}, weight_decay={wd} ---")
                torch.manual_seed(42)
                model = LSTMClassifier(vocab_size=len(vocab), dropout=dropout,
                                       hidden_dim=hidden).to(device)
                settings = {"dropout": dropout, "hidden_dim": hidden}
                run_one(model, 1e-3, epochs, settings, results, best,
                        weight_decay=wd)

    return save_results(results, best, "lstm")



# RQ1: Identify which LLM generated the response

# CNN testing data
cnn, cnn_history = tune_cnn()
plot_history(cnn_history, "CNN", "cnn_curves.png")
test_loss, test_acc, preds, labels = evaluate(cnn, test_loader)
print(f"CNN test accuracy: {test_acc:.3f}")
print(f"CNN test macro-F1: {f1_score(labels, preds, average='macro'):.3f}")
print(classification_report(labels, preds, target_names=le.classes_))
print(confusion_matrix(labels, preds))

# LSTM testing data
lstm, lstm_history = tune_lstm()
plot_history(lstm_history, "LSTM", "lstm_curves.png")
test_loss, test_acc, preds, labels = evaluate(lstm, test_loader)
print(f"\nLSTM test accuracy: {test_acc:.3f}")
print(f"LSTM test macro-F1: {f1_score(labels, preds, average='macro'):.3f}")
print(classification_report(labels, preds, target_names=le.classes_))
print(confusion_matrix(labels, preds))
