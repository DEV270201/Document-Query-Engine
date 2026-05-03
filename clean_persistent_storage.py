from pathlib import Path as path
cwd = path.cwd()
PERSISTENT_DIR=fr'{cwd}\db\resume'

while True:
    file_name = input("Enter the storage file that you want to clean: ")
    if not file_name.strip():
        print("No name provided | Aborting .....")
        continue
    file_path = path(f"{PERSISTENT_DIR}\{file_name.strip()}")
    if not file_path.is_file():
        print("Either the file does not exists or it is not a file | Aborting ....")
        continue

    with open(file_path,'w',encoding='utf-8') as f:
        f.write("")
    
    print("Persistent storage cleaned ....")
    break








