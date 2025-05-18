import os

def sync_with_github():
    os.system("git add .")
    os.system("git commit -m 'Auto sync after training'")
    os.system("git push origin main")

if __name__ == "__main__":
    sync_with_github()
