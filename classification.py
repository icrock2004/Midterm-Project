import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    cols = ["llm_name", "llm_input", "llm_output"]

    with open(path, encoding="utf-8") as f:
        raw = f.read()
    records = []

    for block in raw.split():
        if not block.strip():
            continue
        name, inp, out = block.split()