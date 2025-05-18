import torch
from test_config import layer_name, model_name
from utils.model_loader import load_or_initialize_model
from utils.tokenizer import encode_text, decode_tokens, PAD_ID, EOS_ID

def generate_response(model, input_ids, max_len=50, top_k=10, temperature=1.0):
    device = input_ids.device
    generated = input_ids.clone()  # (1, seq_len)

    for _ in range(max_len):
        with torch.no_grad():
            output = model(generated)[0]

            if output.size(1) == 0:
                print("[HATA]: Model boş çıktı verdi.")
                return [PAD_ID]

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

            # DEBUG: göster
            print(f"[DEBUG] Üretilen token ID: {next_token_id.item()} ({decode_tokens([next_token_id.item()])})")

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

            ids = encode_text(prompt)
            print(f"[DEBUG] Input IDs: {ids}")  # INPUT debug

            if len(ids) == 0:
                print("[AI]: Geçersiz giriş.")
                continue

            input_ids = torch.tensor([ids], dtype=torch.long).to(device)
            predicted_ids = generate_response(model, input_ids)
            response = decode_tokens(predicted_ids)

            print(f"[AI]: {response}")

        except Exception as e:
            print(f"[HATA]: {str(e)}")

if __name__ == "__main__":
    main()
