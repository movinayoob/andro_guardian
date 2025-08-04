import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#Explain the data
df = pd.read_csv("data/processed/output_features_all.csv")
print(df.shape)
print(df["label"].value_counts())
df.head()

#Handle Missing values
df.fillna(0, inplace=True)

#Remove Constant / Low-Variance Columns
X = df.drop("label", axis=1)
print("Total system call features before filtering:", X.shape[1])
df["total_syscalls_per_app"] = X.sum(axis=1)

plt.figure(figsize=(8, 5))
sns.histplot(df["total_syscalls_per_app"], bins=50, kde=True)
plt.title("Distribution of Total Syscalls per App")
plt.xlabel("Total Syscall Count")
plt.ylabel("Number of Applications")
plt.grid(True)
plt.show()

print("Syscall count per app statistics before filtering:")
print(df["total_syscalls_per_app"].describe())
zero_counts = (X == 0).sum()
always_zero = zero_counts[zero_counts == len(X)]
print(f"Number of syscalls never used in any app before filtering: {always_zero.shape[0]}")

selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)



# Rebuild dataframe with selected columns
selected_features = X.columns[selector.get_support()]
df_filtered = pd.DataFrame(X_filtered, columns=selected_features)
df_filtered["label"] = df["label"]
print(df_filtered.shape)
print(df_filtered["label"].value_counts())
df_filtered.head()
print("Total system call features after low-variance filtering:", df_filtered.shape[1] - 1)
df_filtered["total_syscalls_per_app"] = df_filtered.drop("label", axis=1).sum(axis=1)

plt.figure(figsize=(8, 5))
sns.histplot(df_filtered["total_syscalls_per_app"], bins=50, kde=True)
plt.title("Distribution of Total Syscalls per App after low-variance filtering")
plt.xlabel("Total Syscall Count")
plt.ylabel("Number of Applications")
plt.grid(True)
plt.show()

print("Syscall count per app statistics:")
print(df_filtered["total_syscalls_per_app"].describe())
zero_counts = (X_filtered == 0).sum()
always_zero = zero_counts[zero_counts == len(X)]
print(f"Number of syscalls never used in any app: {always_zero.shape[0]}")

nonzero_before = (df.drop("label", axis=1) > 0).sum(axis=1)
nonzero_after = (df_filtered.drop("label", axis=1) > 0).sum(axis=1)

before_total = df.drop("label", axis=1).sum(axis=1)
after_total = df_filtered.drop("label", axis=1).sum(axis=1)

# Compare difference
diff = before_total - after_total
print("Mean difference:", diff.mean())
print("Max difference:", diff.max())
print("Number of apps with any difference:", (diff != 0).sum())

sns.histplot(nonzero_before, color="blue", label="Before", kde=True)
sns.histplot(nonzero_after, color="green", label="After", kde=True)
plt.legend()
plt.title("Non-zero Syscall Types per App (Before vs After)")
plt.xlabel("Syscall Types > 0")
plt.show()

le = LabelEncoder()
df_filtered["label"] = le.fit_transform(df_filtered["label"])
# 0 = benign, 1 = malware (for example)

scaler = StandardScaler()
features = df_filtered.drop("label", axis=1)
scaled_features = scaler.fit_transform(features)

df_normalized = pd.DataFrame(scaled_features, columns=features.columns)
df_normalized["label"] = df_filtered["label"]
print(df_normalized.shape)
print(df_normalized.head())

df_normalized.to_csv("data/processed/cleaned_data.csv", index=False)


joblib.dump(scaler, "data/other/feature_scaler.pkl")