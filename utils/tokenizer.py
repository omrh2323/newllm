import sentencepiece as spm

# Tokenizer yükle
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/sp.model")  # Gerekirse path dışarıdan da alınabilir

# Özel token ID'leri
vocab_size = sp.get_piece_size()
PAD_ID = sp.pad_id()
UNK_ID = sp.unk_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()

def encode_text(text, add_bos=True, add_eos=True, max_len=None):
    """
    Metni token ID'lerine çevirir.
    BOS/EOS opsiyoneldir. max_len varsa içerik sınırlandırılır.
    """
    ids = sp.encode(text, out_type=int)

    if max_len:
        max_body_len = max_len - int(add_bos) - int(add_eos)
        ids = ids[:max_body_len]

    if add_bos:
        ids = [BOS_ID] + ids
    if add_eos:
        ids.append(EOS_ID)

    return ids

def decode_tokens(token_ids):
    """
    Token ID listesini metne çevirir.
    PAD, BOS, EOS, UNK gibi özel tokenlar filtrelenir.
    """
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    elif not isinstance(token_ids, list):
        try:
            token_ids = list(token_ids)
        except Exception:
            return "[Hatalı token giriş]"

    filtered = [
        t for t in token_ids
        if 0 <= t < vocab_size and t not in [PAD_ID, BOS_ID, EOS_ID, UNK_ID]
    ]

    if not filtered:
        return "[Boş çıktı]"

    return sp.decode(filtered)
