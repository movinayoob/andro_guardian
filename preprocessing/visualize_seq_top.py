import json
from collections import Counter
import matplotlib.pyplot as plt

# Load tokenizer JSON
with open("coverage_tokenizer_syscalls_cumulative_28.json", "r") as f:
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