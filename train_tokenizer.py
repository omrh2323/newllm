import os
import sentencepiece as spm

def train_general_tokenizer():
    input_file = "data/intent_data.txt"
    output_dir = os.path.abspath("tokenizer")
    os.makedirs(output_dir, exist_ok=True)

    model_prefix = os.path.join(output_dir, "sp_general")  # 🔁 özel prefix

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=1000,
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

    print(f"[✓] Genel tokenizer başarıyla üretildi: {model_prefix}.model")

if __name__ == "__main__":
    train_general_tokenizer()
