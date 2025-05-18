import os
import sentencepiece as spm

# Tokenizer yükleniyor (IntentAI için sabit path)
TOKENIZER_PATH = "intent_tokenizer/intent_sp.model"

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"[HATA] Tokenizer modeli bulunamadı: {TOKENIZER_PATH}")

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

# Özel token ID'leri
vocab_size = sp.get_piece_size()
PAD_ID = sp.pad_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()
UNK_ID = sp.unk_id()

def encode_text(text, add_bos=True, add_eos=True):
    """
    Metni token ID'lerine çevirir.
    BOS/EOS eklemek isteğe bağlıdır.
    """
    ids = sp.encode(text, out_type=int)
    if add_bos:
        ids = [BOS_ID] + ids
    if add_eos:
        ids.append(EOS_ID)
    return ids

def decode_tokens(token_ids):
    """
    Token ID listesini tekrar metne çevirir.
    Özel token'lar filtrelenir.
    """
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    elif not isinstance(token_ids, list):
        try:
            token_ids = list(token_ids)
        except Exception:
            return "[Geçersiz token]"

    filtered = [
        t for t in token_ids
        if 0 <= t < vocab_size and t not in [PAD_ID, BOS_ID, EOS_ID, UNK_ID]
    ]
    return sp.decode(filtered) if filtered else "[Boş]"
