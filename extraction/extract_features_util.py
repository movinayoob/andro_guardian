import os
import random
import re

malware_path = "data/raw/Malware/systemcalls"
benign_path = "data/raw/Benign/systemcalls/systemcalls"
# -------------------------------
# Get file paths
# -------------------------------
def get_file_paths(folder, label, limit):
    all_files = os.listdir(folder)
    random.shuffle(all_files)  # Shuffle in place
    selected_files = all_files[:limit]  # Take only the first 'limit' files
    return [(os.path.join(folder, f), label) for f in selected_files]

# -------------------------------
# Extract syscall names
# -------------------------------
def read_syscall_sequence(file_path):
    syscall_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]+)\(')
    sequence = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = syscall_pattern.match(line.strip())
                if match:
                    sequence.append(match.group(1))
    except UnicodeDecodeError:
        print(f"⚠️ Skipping unreadable file: {file_path}")
    return sequence

# -------------------------------
# Enforce Vocabulary Limit
# -------------------------------
def enforce_vocab_limit(sequences, vocab_size):
    return [[token if token < vocab_size else 1  # Replace with OOV index
             for token in seq] for seq in sequences]