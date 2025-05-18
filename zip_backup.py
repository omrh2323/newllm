import zipfile
import os

# Zip adı kullanıcıdan alınır
zip_name = input("Zip dosyasının adını gir (örnek: yedek_01): ").strip()
if not zip_name.endswith(".zip"):
    zip_name += ".zip"

# Yedeklenecek uzantılar ve klasörler
include_extensions = [".py", ".pt", ".model", ".vocab", ".txt"]
include_dirs = ["models", "tokenizer", "intent_tokenizer", "utils", "data"]

# Zip dosyasını oluştur
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk("."):
        if any(skip in root for skip in [".venv", "__pycache__"]):
            continue
        if not any(root.startswith(f".{os.sep}{inc}") for inc in include_dirs) and root != ".":
            continue
        for file in files:
            if any(file.endswith(ext) for ext in include_extensions):
                file_path = os.path.join(root, file)
                zipf.write(file_path)

print(f"[✓] Yedek başarıyla oluşturuldu: {zip_name}")
