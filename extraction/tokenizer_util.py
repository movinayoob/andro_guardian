import json
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from extract_features_util import read_syscall_sequence,enforce_vocab_limit
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# -------------------------------
# Save Tokenizer to Json file
# -------------------------------
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

# -------------------------------
# Load Tokenizer
# -------------------------------
def load_tokenizer(filepath):
    with open(filepath, "r") as f:
        json_dict = json.load(f)
    tokenizer_json = json.dumps(json_dict)  # re-convert to string
    return tokenizer_from_json(tokenizer_json)
""" 
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
    return df_batch """

# -------------------------------
# Function to process batch
# -------------------------------
def process_batch(batch_files, tokenizer, max_len=None):
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

    if max_len is None:
        max_len = max(len(seq) for seq in sequences_tokenized) if sequences_tokenized else 0

    sequences_padded = pad_sequences(sequences_tokenized, maxlen=max_len, padding='post', truncating='post')
    df_batch = pd.DataFrame(sequences_padded)
    df_batch['label'] = batch_labels
    df_batch.insert(0, 'label', df_batch.pop('label'))
    return df_batch

def enforce_whitelist(tokenizer, whitelist_syscalls):
    for syscall in whitelist_syscalls:
        if syscall not in tokenizer.word_index:
            tokenizer.word_index[syscall] = len(tokenizer.word_index) + 1

    whitelist_indices = [tokenizer.word_index[s] for s in whitelist_syscalls if s in tokenizer.word_index]
    tokenizer.num_words = max(tokenizer.num_words or 0, max(whitelist_indices) + 1)
    print(f"✅ Enforced whitelist. New num_words: {tokenizer.num_words}")
    return tokenizer