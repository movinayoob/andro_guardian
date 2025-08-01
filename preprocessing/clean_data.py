import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib

#Explain the data
df = pd.read_csv("data/processed/output_features_all.csv")
print(df.shape)
print(df["label"].value_counts())
df.head()

#Handle Missing values
df.fillna(0, inplace=True)

#Remove Constant / Low-Variance Columns
X = df.drop("label", axis=1)
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)

# Rebuild dataframe with selected columns
selected_features = X.columns[selector.get_support()]
df_filtered = pd.DataFrame(X_filtered, columns=selected_features)
df_filtered["label"] = df["label"]
print(df_filtered.shape)
print(df_filtered["label"].value_counts())
df_filtered.head()

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