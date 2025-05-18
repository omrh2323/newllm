import torch
from test_config import layer_name, model_name
from utils.model_loader import load_or_initialize_model
from train_settings import tokenizer_type
from utils.tokenizer import encode_text, decode_tokens, PAD_ID, EOS_ID
from utils.intent_tokenizer import encode_text as intent_encode, decode_tokens as intent_decode, PAD_ID as intent_PAD_ID, EOS_ID as intent_EOS_ID

def generate_response(model, input_ids, max_len=50, top_k=5, temperature=1.0):
    device = input_ids.device
    generated = input_ids.clone()

    for _ in range(max_len):
        with torch.no_grad():
            output = model(generated)

            if output.dim() == 2:
                last_logits = output[-1] / temperature
            elif output.dim() == 3:
                last_logits = output[:, -1, :] / temperature
                last_logits = last_logits.squeeze(0)
            else:
                raise ValueError(f"[!] Geçersiz çıktı boyutu: {output.shape}")

            probs = torch.softmax(last_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
            next_token_idx = torch.multinomial(top_k_probs, 1).item()
            next_token_id = top_k_indices[next_token_idx]

        if next_token_id.item() == EOS_ID:
            break

        next_token_id = next_token_id.unsqueeze(0).unsqueeze(0).to(device)
        generated = torch.cat((generated, next_token_id), dim=1)

    return generated[0].tolist()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_or_initialize_model(layer_name, model_name)
    model.to(device)
    model.eval()

    print(f"[✓] {layer_name}/{model_name} yüklendi. Teste hazırsın!")

    while True:
        try:
            prompt = input("\n[Sen]: ").strip()
            if prompt.lower() in ["exit", "quit"]:
                break

            # Tokenizer'ı türüne göre seç
            if tokenizer_type == "intent":
                input_ids = torch.tensor([intent_encode(prompt)], dtype=torch.long).to(device)
                decode_fn = intent_decode
            else:
                input_ids = torch.tensor([encode_text(prompt)], dtype=torch.long).to(device)
                decode_fn = decode_tokens

            if input_ids.size(1) == 0:
                print("[AI]: Geçersiz giriş.")
                continue

            predicted_ids = generate_response(model, input_ids)
            response = decode_fn(predicted_ids)

            print(f"[AI]: {response}")

        except Exception as e:
            print(f"[HATA]: {str(e)}")

if __name__ == "__main__":
    main()