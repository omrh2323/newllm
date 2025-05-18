layer_name = "intent_classifier"              # Katman adı (örneğin intent sınıflayıcı)
model_name = "tinyllm_intent"                 # Modelin kayıt klasör/adı
data_path = "data/intent_verisi.txt"          # Eğitim verisi yolu

epochs = 3                                     # Epoch sayısı
batch_size = 128                               # Aynı anda eğitilecek örnek sayısı
lr = 5e-5                                       # Learning rate (öğrenme hızı)

# Tokenizer tipi: "intent" -> intent_tokenizer, "default" -> genel tokenizer
tokenizer_type = "intent"
