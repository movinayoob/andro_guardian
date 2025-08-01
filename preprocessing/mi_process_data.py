import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import LabelEncoder,KBinsDiscretizer
import joblib

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

discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X_binned_array = discretizer.fit_transform(X_encoded)

# Convert back to DataFrame to retain column names
X_binned = pd.DataFrame(X_binned_array, columns=X_encoded.columns)

# Mutual Information-based Feature Selection
#X_mi = X_binned.drop("label", axis=1)
#y = X_binned["label"]

# Compute MI scores
mi_scores = mutual_info_classif(X_binned, y_encoded)
mi_series = pd.Series(mi_scores, index=X_binned.columns)
mi_series_sorted = mi_series.sort_values(ascending=False)

# Select top K features based on MI (optional: change K)
K = 120  # you can choose a different value based on your dataset size
top_features = mi_series_sorted.head(K).index.tolist()

# Filter DataFrame to top MI features
df_mi_selected = df_filtered[top_features + ["label"]]
print("Final shape after MI selection:", df_mi_selected.shape)

# Save final features to CSV
df_mi_selected.to_csv("data/processed/feature_mi_selected_120.csv", index=False)
print("Saved features to features_mi_selected.csv")