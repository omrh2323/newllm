import sys, os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from train_settings import layer_name, model_name, data_path, epochs, batch_size, lr, tokenizer_type
from utils.model_loader import load_or_initialize_model
from utils.data_loader import load_training_data
from torch.nn.utils.rnn import pad_sequence

# Tokenizer tipi kontrolü
if tokenizer_type == "intent":
    from utils.intent_tokenizer import encode_text, PAD_ID
else:
    from utils.tokenizer import encode_text, PAD_ID

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modeli yükle veya oluştur
    model, model_path = load_or_initialize_model(layer_name, model_name)
    model.to(device)
    model.train()

    # Veriyi yükle
    dataloader = load_training_data(data_path, batch_size=batch_size, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    scaler = GradScaler()

    # Checkpoint dizini
    checkpoint_dir = os.path.join("checkpoints", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0

        for step, (questions, answers) in enumerate(dataloader):
            q_ids_batch = [encode_text(q, add_bos=True, add_eos=True) for q in questions]
            a_ids_batch = [encode_text(a, add_bos=False, add_eos=False) for a in answers]

            # Boş veya hatalı örnekleri filtrele
            filtered = [(q, a) for q, a in zip(q_ids_batch, a_ids_batch)
                        if isinstance(q, list) and isinstance(a, list) and len(q) > 0 and len(a) > 0]
            if len(filtered) == 0:
                continue

            q_ids_batch, a_ids_batch = zip(*filtered)

            input_ids = pad_sequence([torch.tensor(q, dtype=torch.long) for q in q_ids_batch],
                                     batch_first=True, padding_value=PAD_ID).to(device)
            target_ids = pad_sequence([torch.tensor(a, dtype=torch.long) for a in a_ids_batch],
                                      batch_first=True, padding_value=PAD_ID).to(device)

            optimizer.zero_grad()

            with autocast():
                output = model(input_ids)[0]  # (batch_size, seq_len, vocab_size)
                output = output.view(-1, output.size(-1))
                target = target_ids.view(-1)

                min_len = min(output.size(0), target.size(0))
                output = output[:min_len]
                target = target[:min_len]

                loss = loss_fn(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"[Epoch {epoch+1}] Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"==> Epoch {epoch+1}/{epochs} tamamlandı | Ortalama Loss: {avg_loss:.4f}")

        # Checkpoint kaydı
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[✓] Checkpoint kaydedildi: {checkpoint_path}")

    # Final modeli kaydet
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[✓] Final model kaydedildi: {model_path}")

    # Eğitim sonrası github push (manuel çalıştırılmalı komut satırından)
    os.system("git add .")
    os.system("git commit -m 'Auto training update'")
    os.system("git push origin main")

if __name__ == "__main__":
    train()
