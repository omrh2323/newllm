import os
from torch.utils.data import Dataset, DataLoader

class IntentDataset(Dataset):
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"[HATA] Veri dosyası bulunamadı: {data_path}")

        self.examples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line or "soru:" not in line or "cevap:" not in line:
                    continue
                try:
                    parts = line.split("cevap:")
                    question = parts[0].replace("soru:", "").strip()
                    answer = parts[1].strip()
                    if question and answer:
                        self.examples.append((question, answer))
                except Exception as e:
                    print(f"[!] Satır atlandı (satır {line_no}): {line} | Hata: {e}")

        if not self.examples:
            raise ValueError("[HATA] Yüklenen veri boş, uygun satır bulunamadı.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def load_training_data(data_path, batch_size=64, num_workers=2, pin_memory=True):
    dataset = IntentDataset(data_path)

    # Gelişmiş kontrol: batch_size büyükse, worker sayısını artır
    if batch_size >= 128 and num_workers < 4:
        num_workers = 4

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader
