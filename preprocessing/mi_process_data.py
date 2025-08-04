import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import LabelEncoder,KBinsDiscretizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("data/processed/output_features_all.csv")
print("Original shape:", df.shape)
print("Label distribution:\n", df["label"].value_counts())
print(df.head())

# Handle Missing Values
df.fillna(0, inplace=True)

# Remove Constant / Low-Variance Columns
X = df.drop("label", axis=1)
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)

# Rebuild dataframe with selected columns
selected_features = X.columns[selector.get_support()]
df_filtered = pd.DataFrame(X_filtered, columns=selected_features)
df_filtered["label"] = df["label"]
print("After low-variance filter:", df_filtered.shape)

# Label Encoding (if 'label' is categorical)
le = LabelEncoder()
df_filtered["label"] = le.fit_transform(df_filtered["label"])
# 0 = benign, 1 = malware (example)

# Separate features and target
X_encoded = df_filtered.drop("label", axis=1)
y_encoded = df_filtered["label"]

#discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')
#X_binned_array = discretizer.fit_transform(X_encoded)

# Convert back to DataFrame to retain column names
#X_binned = pd.DataFrame(X_binned_array, columns=X_encoded.columns)

# Mutual Information-based Feature Selection
#X_mi = X_binned.drop("label", axis=1)
#y = X_binned["label"]

# Compute MI scores
mi_scores = mutual_info_classif(X_encoded, y_encoded, discrete_features=False)
mi_series = pd.Series(mi_scores, index=X_encoded.columns)
mi_series_sorted = mi_series.sort_values(ascending=False)

# Select top K features based on MI (optional: change K)
K = 120  # you can choose a different value based on your dataset size
top_features = mi_series_sorted.head(K).index.tolist()

# Filter DataFrame to top MI features
df_mi_selected = df_filtered[top_features + ["label"]]
print("Final shape after MI selection:", df_mi_selected.shape)

# Save final features to CSV
df_mi_selected.to_csv("data/processed/feature_mi_selected_120_5bin.csv", index=False)
# Bar plot of top 30 features by MI score
top_mi = mi_series_sorted.head(30)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_mi.values, y=top_mi.index, palette="viridis")
plt.xlabel("Mutual Information Score")
plt.ylabel("Feature")
plt.title("Top 30 Features by Mutual Information Score")
plt.tight_layout()
plt.savefig("new_figures/mi_top30_features.png", dpi=300)
plt.show()

X_mi = df_mi_selected.drop("label", axis=1)
y_mi = df_mi_selected["label"]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_mi)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_mi, palette=["skyblue", "salmon"], alpha=0.5)
plt.title("PCA Projection Using Top 120 MI-Selected Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Class", labels=["Benign", "Malware"])
plt.tight_layout()
plt.savefig("new_figures/pca_mi_features.png", dpi=300)
plt.show()

before_mi_count = X_encoded.shape[1]
after_mi_count = len(top_features)

plt.figure(figsize=(6, 4))
sns.barplot(x=["Before MI", "After MI"], y=[before_mi_count, after_mi_count], palette="muted")
plt.title("Feature Count Before and After MI Selection")
plt.ylabel("Number of Features")
plt.tight_layout()
plt.savefig("new_figures/feature_count_reduction.png", dpi=300)
plt.show()
print("Saved features to features_mi_selected.csv")