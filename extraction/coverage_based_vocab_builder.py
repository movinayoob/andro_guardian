import argparse
import random
from extract_features_util import get_file_paths,read_syscall_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tokenizer_util import save_tokenizer,load_tokenizer,process_batch,enforce_whitelist
import os
# -------------------------------
# ARGPARSE CONFIGURATION
# -------------------------------
parser = argparse.ArgumentParser(description="Syscall sequence tokenizer and saver")

parser.add_argument("--batch_size", type=int, default=500, help="Number of files per batch")
parser.add_argument("--max_len", type=int, required=False , help="Maximum sequence length")
parser.add_argument("--limit", type=int, default=5000, help = "Maximum number of files from each class")
parser.add_argument("--vocab_file", type=str, default="coverage_tokenizer_syscalls_cumulative_", help="Tokenizer output path")
parser.add_argument("--output_csv", type=str, default="data/processed/covearge_sequential_syscall_", help="Output CSV path")

args = parser.parse_args()

# Assign to variables for convenience
BATCH_SIZE = args.batch_size
MAX_LEN = args.max_len
VOCAB_FILE = args.vocab_file
OUTPUT_CSV = args.output_csv
LIMIT = args.limit

malware_path = "data/raw/Malware/systemcalls"
benign_path = "data/raw/Benign/systemcalls/systemcalls"

malware_files = get_file_paths(malware_path, 1, LIMIT)
benign_files = get_file_paths(benign_path, 0, LIMIT)

all_files = malware_files + benign_files
random.shuffle(all_files)  # After combining benign and malware

# -------------------------------
# Create tokenizer from a small sample
# -------------------------------
WHITELIST_SYSCALLS = [
    # Security and Privilege Escalation
    'setuid', 'setgid', 'setreuid', 'setregid', 'seteuid', 'setegid',
    'setfsuid', 'setfsgid', 'capset', 'capget',

    # File and Directory Manipulation
    'open', 'openat', 'read', 'write', 'close', 'lseek', 'stat', 'fstat',
    'lstat', 'chmod', 'fchmod', 'chown', 'fchown', 'mkdir', 'rmdir',
    'unlink', 'rename', 'link', 'symlink', 'readlink', 'truncate',
    'ftruncate',

    # Networking
    'socket', 'connect', 'bind', 'listen', 'accept', 'sendto', 'recvfrom',
    'sendmsg', 'recvmsg', 'setsockopt', 'getsockopt', 'shutdown',
    'getpeername', 'getsockname',

    # Process and Memory Manipulation
    'fork', 'vfork', 'clone', 'execve', 'wait4', 'waitpid', 'exit', 'exit_group',
    'kill', 'tgkill', 'mmap', 'mmap2', 'munmap', 'mprotect', 'brk', 'arch_prctl',

    # Inter-Process Communication
    'pipe', 'pipe2', 'dup', 'dup2', 'dup3', 'shmget', 'shmat', 'shmdt',
    'shmctl', 'semget', 'semop', 'semctl', 'msgget', 'msgsnd', 'msgrcv',
    'msgctl',

    # Audit and Control
    'ptrace', 'auditctl', 'prctl', 'setrlimit', 'getrlimit', 'getrusage',
    'syslog', 'personality', 'reboot', 'init_module', 'delete_module'
]
print("📥 Fitting tokenizer on a sample...")
sample_sequences = []
for file_path, _ in all_files[:2000]:
    sample_sequences.append(' '.join(read_syscall_sequence(file_path)))

#tokenizer = Tokenizer(lower=True, oov_token="<OOV>")
#tokenizer.fit_on_texts(sample_sequences)
#tokenizer = enforce_whitelist(tokenizer, WHITELIST_SYSCALLS)

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

# Combine whitelist with top-N
top_syscalls = [word for word, _ in freq_dist.most_common(num_words)]
final_vocab = list(set(top_syscalls).union(set(WHITELIST_SYSCALLS)))
num_words = len(final_vocab) + 1  # +1 to account for OOV token
# Filter sequences to only include syscalls in final vocab
filtered_sequences = [' '.join([word for word in seq.split() if word in final_vocab]) for seq in sample_sequences]

# Fit tokenizer
tokenizer = Tokenizer(lower=True, oov_token="<OOV>")
tokenizer.fit_on_texts(filtered_sequences)
tokenizer.num_words = num_words  # manuall


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

save_tokenizer(tokenizer, f"{VOCAB_FILE}{num_words}.json")

# Load tokenizer
tokenizer = load_tokenizer(f"{VOCAB_FILE}{num_words}.json")
index_to_syscall = {v: k for k, v in tokenizer.word_index.items()}

for i in range(1, 201):  # skip 0 if it's reserved for padding
    syscall_name = index_to_syscall.get(i, "<PAD/UNK>")
    print(f"Column {i}: {syscall_name}")

print("✅ Tokenizer fitted and saved.")
print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")

# -------------------------------
# Process in batches
# -------------------------------
print("🚀 Processing in batches...")
os.makedirs(os.path.dirname(f"{OUTPUT_CSV}{num_words}.csv"), exist_ok=True)

for i in range(0, len(all_files), BATCH_SIZE):
    batch_files = all_files[i:i + BATCH_SIZE]
    print(f"🔄 Processing files {i} to {i + len(batch_files) - 1}")
    if MAX_LEN is not None and MAX_LEN > 0:
        df_batch = process_batch(batch_files, tokenizer, MAX_LEN)
    else:
        df_batch = process_batch(batch_files, tokenizer)

    if i == 0:
        df_batch.to_csv(f"{OUTPUT_CSV}{num_words}.csv", index=False, mode='w')
    else:
        df_batch.to_csv(f"{OUTPUT_CSV}{num_words}.csv", index=False, header=False, mode='a')

print("✅ All batches processed and saved to CSV.")
