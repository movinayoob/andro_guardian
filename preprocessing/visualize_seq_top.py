import json
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# Load tokenizer JSON
with open("coverage_tokenizer_syscalls_cumulative_40k_400_107.json", "r") as f:
    tokenizer_json = json.load(f)

# Extract and parse word_counts
word_counts = json.loads(tokenizer_json['config']['word_counts'])
word_counter = Counter(word_counts)

# Get top-N syscalls
top_n = 30
top_syscalls = word_counter.most_common(top_n)

# Separate names and counts
syscall_names, frequencies = zip(*top_syscalls)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(syscall_names, frequencies)
plt.xticks(rotation=45, ha='right')
plt.title(f"Top {top_n} Most Frequent Syscalls")
plt.xlabel("Syscall")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# Load your syscall sequences (assuming CSV with one row per app and syscalls in a list or string format)
df = pd.read_csv("data/processed/covearge_sequential_syscall_40k_400_107.csv")  # Adjust path as needed
print(df.columns)


# Assuming columns 0 to 399 hold the sequence (adjust if needed)
sequence_columns = [str(i) for i in range(400)]

# Count non-zero entries per row (assuming 0 is used for padding)
sequence_lengths = df[sequence_columns].apply(lambda row: (row != 0).sum(), axis=1)

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(sequence_lengths, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Syscall Sequence Lengths")
plt.xlabel("Sequence Length (non-padded)")
plt.ylabel("Number of Samples")
plt.grid(True)
plt.tight_layout()
plt.savefig("sequence_length_distribution.png")
plt.show()
min_len = sequence_lengths.min()
max_len = sequence_lengths.max()
mean_len = sequence_lengths.mean()
median_len = sequence_lengths.median()
len_95th = int(sequence_lengths.quantile(0.95))

print(f"Minimum sequence length: {min_len}")
print(f"Maximum sequence length: {max_len}")
print(f"Mean sequence length: {mean_len:.2f}")
print(f"Median sequence length: {median_len}")
print(f"95th percentile sequence length (recommended maxlen): {len_95th}")
max_len = int(sequence_lengths.quantile(0.95))  # Use 95th percentile
print(f"95% of sequences are <= {max_len} steps long.")
# Assuming sequences are stored as space-separated strings in a column named 'sequence'
sequence_lengths = df['sequence'].apply(lambda x: len(str(x).split()))

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(sequence_lengths, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Syscall Sequence Lengths")
plt.xlabel("Sequence Length")
plt.ylabel("Number of Samples")
plt.grid(True)
plt.tight_layout()
plt.savefig("sequence_length_distribution.png")
plt.show()