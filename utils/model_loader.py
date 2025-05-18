# utils/model_loader.py

import os
import torch
from torch import nn
from train_settings import tokenizer_type

# Tokenizer'a göre vocab boyutunu içe aktar
if tokenizer_type == "intent":
    from utils.intent_tokenizer import vocab_size
else:
    from utils.tokenizer import vocab_size

class TinyLLM(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

def load_or_initialize_model(layer_name, model_name):
    path = f"models/{layer_name}/{model_name}.pt"
    os.makedirs(f"models/{layer_name}", exist_ok=True)

    model = TinyLLM()

    if os.path.exists(path):
        print(f"[INFO] Mevcut model bulundu, yükleniyor: {path}")
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        print(f"[INFO] Yeni model oluşturuluyor: {path}")

    return model, path
