# demo.py

import torch
from train_settings import tokenizer_type
from utils.model_loader import load_or_initialize_model

# === AYARLAR ===
layer_name = "intent_classifier"
model_name = "tinyllm_intent"
model_epoch_path = f"checkpoints/{model_name}/epoch_1.pt"

# Tokenizer seçimi
if tokenizer_type == "intent":
    from utils.intent_tokenizer import encode_text, sp
else:
    from utils.tokenizer import encode_text, sp

def decode_token(token_id):
    print(f"[DEBUG] Tahmin edilen token ID: {token_id}")
    token = sp.id_to_piece(token_id)
    return token.replace("▁", "").strip()

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_or_initialize_model(layer_name, model_name)
    model.load_state_dict(torch.load(model_epoch_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict(model, device, text):
    input_ids = encode_text(text, add_bos=True, add_eos=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)[0]
        pred_id = torch.argmax(output[0], dim=-1).item()

    return decode_token(pred_id)

if __name__ == "__main__":
    model, device = load_model()
    print("[✓] Model yüklendi. Soru sormaya başlayabilirsin. (çıkmak için: exit / quit)\n")

    while True:
        try:
            text = input("Soru: ")
            if text.strip().lower() in ["exit", "quit", "çık", "q"]:
                break

            result = predict(model, device, text)
            print(f"[Tahmin Edilen Katman] => {result}\n")

        except KeyboardInterrupt:
            print("\n[!] Program durduruldu.")
            break
        except Exception as e:
            print(f"[HATA]: {e}")
