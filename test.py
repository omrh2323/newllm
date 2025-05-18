import os
import torch
from test_config import layer_name, model_name
from utils.model_loader import load_or_initialize_model
from train_settings import tokenizer_type

# Tokenizer tipi seçimi
if tokenizer_type == "intent":
    from utils.intent_tokenizer import encode_text, decode_tokens, PAD_ID, EOS_ID
else:
    from utils.tokenizer import encode_text, decode_tokens, PAD_ID, EOS_ID

def generate_response(model, input_ids, max_len=50, top_k=5, temperature=1.0):
    device = input_ids.device
    model.eval()
    generated = input_ids.clone()

    for _ in range(max_len):
        with torch.no_grad():
            output = model(generated)[0]  # (batch_size, seq_len, vocab_size)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
            next_token = torch.multinomial(top_k_probs, 1)
            next_token_id = top_k_indices.gather(1, next_token).squeeze(1)

        if next_token_id.item() == EOS_ID:
            break

        next_token_id = next_token_id.unsqueeze(0)
        generated = torch.cat((generated, next_token_id), dim=1)

    return generated[0].tolist()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_or_initialize_model(layer_name, model_name)
    model.to(device)

    print(f"[✓] Model yüklendi: {layer_name}/{model_name}")
    print("[!] Çıkmak için: exit")

    while True:
        try:
            prompt = input("\n[Sen]: ")
            if prompt.lower() in ["exit", "quit"]:
                break

            input_ids = torch.tensor([encode_text(prompt)], dtype=torch.long).to(device)
            if input_ids.size(1) == 0:
                print("[AI]: Boş mesaj.")
                continue

            predicted_ids = generate_response(model, input_ids)
            response = decode_tokens(predicted_ids)
            print(f"[AI]: {response}")

        except Exception as e:
            print(f"[HATA]: {str(e)}")

if __name__ == "__main__":
    main()
