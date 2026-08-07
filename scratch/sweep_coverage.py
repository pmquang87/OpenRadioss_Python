import json
import os
import subprocess
import zipfile
import shutil
import sys
import glob
from concurrent.futures import ThreadPoolExecutor

inventory_file = r"C:\Users\pmqua\PycharmProjects\OpenRadioss_Python\tools\validation_data\inventory.json"
inventory = json.load(open(inventory_file))

out_dir = r"C:\temp\coverage_sweep"
os.makedirs(out_dir, exist_ok=True)
decks_dir = os.path.join(out_dir, "decks")
os.makedirs(decks_dir, exist_ok=True)

print("Extracting zips and copying dirs...")
decks_to_run = []
for idx, case in enumerate(inventory["cases"]):
    if not case.get("runnable"):
        continue
    source_path = case["source"]
    case_id = case["case_id"]
    parts = case_id.split("/")
    package = parts[1]
    rel_path = "/".join(parts[2:])
    
    dest_dir = os.path.join(decks_dir, f"{idx}_{package}")
    if not os.path.exists(dest_dir):
        if os.path.isdir(source_path):
            shutil.copytree(source_path, dest_dir)
        else:
            try:
                with zipfile.ZipFile(source_path, 'r') as z:
                    z.extractall(dest_dir)
            except Exception as e:
                print(f"Failed to extract {source_path}: {e}")
                continue
                
    actual_deck_path = os.path.join(dest_dir, rel_path)
    if not os.path.exists(actual_deck_path):
        matches = glob.glob(os.path.join(dest_dir, "**", os.path.basename(rel_path)), recursive=True)
        if matches:
            actual_deck_path = matches[0]
        else:
            print(f"NOT FOUND: {actual_deck_path}")
            continue
            
    decks_to_run.append((idx, actual_deck_path))

print(f"Found {len(decks_to_run)} decks to run")

workdir_base = os.path.join(out_dir, "workdirs")
os.makedirs(workdir_base, exist_ok=True)
validate_script = r"C:\Users\pmqua\PycharmProjects\OpenRadioss_Python\tools\validate_vs_fortran.py"
python_exe = r"C:\Users\pmqua\PycharmProjects\OpenRadioss_Python\.venv\Scripts\python.exe"

def run_deck(item):
    idx, deck = item
    wd = os.path.join(workdir_base, f"wd_{idx}")
    cmd = [python_exe, validate_script, "coverage", "--workdir", wd, deck]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    name = os.path.splitext(os.path.basename(deck))[0]
    res_file = os.path.join(wd, f"coverage_{name}.json")
    if not os.path.exists(res_file):
        return deck, "CRASH", None
        
    data = json.load(open(res_file))
    rc = data.get("starter_rc", -1)
    if rc == -9:
        return deck, "TIMEOUT", None
        
    tail = data.get("tail", "")
    is_crash = "Traceback" in tail or rc not in (0, 1)
    if rc != 0 and "Traceback" in tail:
        is_crash = True
        
    has_error = False
    has_skips = False
    
    for row in data.get("rows", []):
        msg = row.get("message", "")
        if "** ERROR" in msg:
            has_error = True
        if "** WARNING" in msg and "unsupported" in msg.lower():
            has_skips = True
        if row.get("dispatch") == "UNKNOWN":
            has_skips = True
            
    for msgs in data.get("other_messages", {}).values():
        for msg in msgs:
            if "** ERROR" in msg:
                has_error = True
            if "** WARNING" in msg:
                has_skips = True

    if is_crash:
        return deck, "CRASH", tail
    elif has_error or rc != 0:
        return deck, "ERROR", None
    elif has_skips:
        return deck, "SKIPS", None
    else:
        return deck, "CLEAN", None

print("Running coverage...")
stats = {"CLEAN": 0, "SKIPS": 0, "ERROR": 0, "CRASH": 0, "TIMEOUT": 0}
crashes = []

with ThreadPoolExecutor(max_workers=8) as exc:
    for deck, status, tail in exc.map(run_deck, decks_to_run):
        stats[status] += 1
        if status == "CRASH":
            crashes.append((deck, tail))

print(f"\nFinal stats: {stats['CLEAN']} CLEAN / {stats['SKIPS']} SKIPS / {stats['ERROR']} ERROR / {stats['CRASH']} CRASH / {stats['TIMEOUT']} TIMEOUT")

if crashes:
    print("\nCrashes:")
    for deck, tail in crashes:
        print(f"--- {deck} ---")
        if tail:
            print(tail[-1500:])
        else:
            print("No output/JSON")
