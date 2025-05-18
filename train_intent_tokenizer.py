import os
import sentencepiece as spm

def train_intent_tokenizer():
    output_dir = os.path.abspath("intent_tokenizer")
    os.makedirs(output_dir, exist_ok=True)

    model_prefix = os.path.join(output_dir, "intent_sp")
    input_file = "data/intent_data.txt"
    vocab_size = 1000

    if not os.path.exists(input_file):
        print(f"[!] Veri dosyası bulunamadı: {input_file}")
        return

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        eos_piece='[EOS]',
        input_sentence_size=100000,
        shuffle_input_sentence=True,
        hard_vocab_limit=False
    )

    print(f"[✓] Intent tokenizer başarıyla üretildi: {model_prefix}.model")

if __name__ == "__main__":
    train_intent_tokenizer()
