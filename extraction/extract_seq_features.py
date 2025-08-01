import os
import json
import re
import pandas as pd
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# -------------------------------
# CONFIGURATION
# -------------------------------
BATCH_SIZE = 500
MAX_LEN = 200
NUM_WORDS = 20
VOCAB_FILE = "tokenizer_syscalls_cumulative.json"
OUTPUT_CSV = "data/processed/sequential_syscall_40k_400.csv"

malware_path = "data/raw/Malware/systemcalls"
benign_path = "data/raw/Benign/systemcalls/systemcalls"

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

def enforce_vocab_limit(sequences, vocab_size):
    return [[token if token < vocab_size else 1  # Replace with OOV index
             for token in seq] for seq in sequences]

# -------------------------------
# Get file paths
# -------------------------------
def get_file_paths(folder, label):
    all_files = os.listdir(folder)
    random.shuffle(all_files)  # Shuffle in place
    selected_files = all_files[:5500]  # Take only the first 'limit' files
    return [(os.path.join(folder, f), label) for f in selected_files]

malware_files = get_file_paths(malware_path, 1)
benign_files = get_file_paths(benign_path, 0)

all_files = malware_files + benign_files
random.shuffle(all_files)  # After combining benign and malware

# -------------------------------
# Create tokenizer from a small sample
# -------------------------------
print("📥 Fitting tokenizer on a sample...")
sample_sequences = []
for file_path, _ in all_files[:2000]:
    sample_sequences.append(' '.join(read_syscall_sequence(file_path)))

tokenizer = Tokenizer(lower=True, oov_token="<OOV>")
tokenizer.fit_on_texts(sample_sequences)

all_syscalls = ' '.join(sample_sequences).split()

# Step 2: Count syscall frequencies
freq_dist = Counter(all_syscalls)

# Step 3: Sort frequencies in descending order
sorted_freqs = sorted(freq_dist.values(), reverse=True)
total_count = sum(sorted_freqs)

# Step 4: Compute cumulative percentage
cumulative = np.cumsum(sorted_freqs)
coverage = cumulative / total_count

# Step 5: Find the smallest N that covers 95% (or any other threshold)
target_coverage = 0.98
num_words = int(np.argmax(coverage >= target_coverage) + 1 ) # +1 for correct index

print(f"✅ Number of top syscalls needed to cover {int(target_coverage*100)}% of data: {num_words}")

# Step 6: Plot cumulative coverage curve
plt.figure(figsize=(10, 5))
plt.plot(coverage * 100)
plt.axhline(y=target_coverage * 100, color='r', linestyle='--')
plt.axvline(x=num_words, color='g', linestyle='--')
plt.title("Cumulative Coverage of Syscall Frequencies")
plt.xlabel("Number of Top Syscalls")
plt.ylabel("Cumulative Coverage (%)")
plt.grid(True)
plt.show()

print(f"Total sample sequences: {len(sample_sequences)}")
print("First 3 samples:", sample_sequences[:1])

tokenizer = Tokenizer(oov_token="<OOV>", num_words=num_words)
tokenizer.fit_on_texts(sample_sequences)

print(f"Total sample sequences: {len(sample_sequences)}")
print("First 3 samples:", sample_sequences[0])
print("🔢 tokenizer.word_index size:", len(tokenizer.word_index))
print("🔤 Top 10 syscalls:", list(tokenizer.word_index.items())[:10])
print("tokenizer.num_words =", tokenizer.num_words)
#sequences = tokenizer.texts_to_sequences(sample_sequences)
#max_index_used = max([max(seq) for seq in sequences if seq], default=0)
#print(f"✅ Max token index used: {max_index_used} (should be < {num_words})")
def save_tokenizer(tokenizer, filepath):
    # 1. Convert JSON string to dict
    raw_json = tokenizer.to_json()
    json_dict = json.loads(raw_json)

    # 2. Safely convert np.int64 or other numpy types to native Python types
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        else:
            return obj

    clean_json = convert(json_dict)

    # 3. Save to file
    with open(filepath, "w") as f:
        json.dump(clean_json, f)

def load_tokenizer(filepath):
    with open(filepath, "r") as f:
        json_dict = json.load(f)
    tokenizer_json = json.dumps(json_dict)  # re-convert to string
    return tokenizer_from_json(tokenizer_json)

# Save tokenizer
save_tokenizer(tokenizer, "tokenizer_syscalls_11.json")

# Load tokenizer
tokenizer = load_tokenizer("tokenizer_syscalls_11.json")

index_to_syscall = {v: k for k, v in tokenizer.word_index.items()}

for i in range(1, 201):  # skip 0 if it's reserved for padding
    syscall_name = index_to_syscall.get(i, "<PAD/UNK>")
    print(f"Column {i}: {syscall_name}")

print("✅ Tokenizer fitted and saved.")
print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")

# -------------------------------
# Function to process batch
# -------------------------------
def process_batch(batch_files, tokenizer, max_len):
    batch_sequences = []
    batch_labels = []

    for file_path, label in batch_files:
        sequence = read_syscall_sequence(file_path)
        sequence_text = ' '.join(sequence)
        batch_sequences.append(sequence_text)
        batch_labels.append(label)

    sequences_tokenized = tokenizer.texts_to_sequences(batch_sequences)
    sequences_tokenized = enforce_vocab_limit(sequences_tokenized, tokenizer.num_words)
    # Debug: check max token index in this batch
    flattened = [t for seq in sequences_tokenized for t in seq]
    if flattened:
        max_token = max(flattened)
        print(f"🔢 Max token index used in this batch: {max_token}")
        print(f"📌 Should be < NUM_WORDS ({tokenizer.num_words})")
    else:
        print("⚠️ No valid tokens found in this batch")
    
    sequences_padded = pad_sequences(sequences_tokenized, maxlen=max_len, padding='post', truncating='post')
    df_batch = pd.DataFrame(sequences_padded)
    df_batch['label'] = batch_labels
    df_batch.insert(0, 'label', df_batch.pop('label'))
    return df_batch

# -------------------------------
# Process in batches
# -------------------------------
print("🚀 Processing in batches...")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

for i in range(0, len(all_files), BATCH_SIZE):
    batch_files = all_files[i:i + BATCH_SIZE]
    print(f"🔄 Processing files {i} to {i + len(batch_files) - 1}")
    df_batch = process_batch(batch_files, tokenizer, MAX_LEN)

    if i == 0:
        df_batch.to_csv(OUTPUT_CSV, index=False, mode='w')
    else:
        df_batch.to_csv(OUTPUT_CSV, index=False, header=False, mode='a')

print("✅ All batches processed and saved to CSV.")
