import os
import subprocess

# Git ayarları (ilk kez yapılandırma için)
GIT_USERNAME = "omrh2323"
GIT_EMAIL = "omeraybas2008@gmail.com"
REPO_URL = "https://github.com/omrh2323/newllm.git"

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[HATA] Komut başarısız: {command}\n{result.stderr}")
    else:
        print(f"[✓] {command}")
    return result

def setup_git_config():
    run_command(f"git config --global user.name \"{GIT_USERNAME}\"")
    run_command(f"git config --global user.email \"{GIT_EMAIL}\"")

def push_to_github():
    print("[✓] GitHub push işlemi başlatılıyor...")
    setup_git_config()
    run_command("git init")
    run_command(f"git remote add origin {REPO_URL}")
    run_command("git add .")
    run_command("git commit -m \"Auto update after training.\"")
    run_command("git branch -M main")
    run_command("git push -u origin main")

if __name__ == "__main__":
    push_to_github()
