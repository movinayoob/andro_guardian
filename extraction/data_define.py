import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/covearge_sequential_syscall_40k_400_107.csv")  # Replace with your actual path

# Option 1: If 'label' is the target column
num_features = df.shape[1] - 1  # Subtract 1 for the label
print(f"Number of features (excluding label): {num_features}")

# Option 2: If you don't know the label column
# Simply print total columns
print(f"Total columns: {df.shape[1]}")
print(f"Column names: {df.columns.tolist()}")