import os
import torch
from torch import nn

# 🔧 Tokenizer tipi ayar dosyasından alınır
from train_settings import tokenizer_type

# 🔄 Tokenizer'a göre doğru vocab size'ı içe aktar
if tokenizer_type == "intent":
    from utils.intent_tokenizer import vocab_size
else:
    from utils.tokenizer import vocab_size

class TinyLLM(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=12, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.fc(x)
        return x

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
