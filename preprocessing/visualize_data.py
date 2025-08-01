import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def plot_umap_projection():
    X = df.drop("label", axis=1)
    y = df["label"]
    
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette="coolwarm", alpha=0.6)
    plt.title("UMAP Projection (2D)")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.text(
        x=min(X_umap[:, 0]) + 0.5,   # position near the bottom-left
        y=min(X_umap[:, 1]) + 0.5,
        s="Sequential System calls",           # your name or label
        fontsize=10,
        color="gray",
        alpha=0.5
    )
    plt.legend(title="Label")
    plt.savefig("reports/mi_seq_umap_2d_projection.png")
    plt.close()

# Load cleaned dataset
df = pd.read_csv("data/processed/coverage_sequential_syscall_28.csv")

# ----------------------------
# 1. Class Distribution
# ----------------------------
def plot_class_distribution():
    plt.figure(figsize=(6, 4))
    sns.countplot(x="label", data=df)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig("reports/10_seq_class_distribution.png")
    plt.close()

# ----------------------------
# 2. Top N Most Active Syscalls
# ----------------------------
def plot_top_syscalls(n=20):
    syscall_totals = df.drop("label", axis=1).sum().sort_values(ascending=False).head(n)
    plt.figure(figsize=(10, 6))
    syscall_totals.plot(kind="bar")
    plt.title(f"Top {n} Most Frequent Syscalls")
    plt.ylabel("Total Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/10_seq_top_syscalls.png")
    plt.close()

# ----------------------------
# 3. Correlation Heatmap
# ----------------------------
def plot_correlation_heatmap():
    plt.figure(figsize=(12, 10))
    sample_df = df.drop("label", axis=1).sample(n=min(1000, len(df)), axis=0)  # for readability
    corr = sample_df.corr()
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.1)
    plt.title("Feature Correlation Heatmap (Sampled)")
    plt.savefig("reports/10_seq_correlation_heatmap.png")
    plt.close()

# ----------------------------
# 4. PCA 2D Visualization
# ----------------------------
def plot_pca_projection():
    X = df.drop("label", axis=1)
    y = df["label"]
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="coolwarm", alpha=0.6)
    plt.title("PCA Projection (2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Label")
    plt.savefig("reports/10_seq_pca_2d_projection.png")
    plt.close()

# ----------------------------
# 5.T-SNE Visualization
# ----------------------------
def plot_tsne_projection():
    X = df.drop("label", axis=1)
    y = df["label"]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="coolwarm", alpha=0.6)
    plt.title("t-SNE Projection (2D)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Label")
    plt.savefig("reports/10_seq_tsne_2d_projection.png")
    plt.close()

# ----------------------------
# 5.UMAP Visualization
# ----------------------------
def plot_umap_projection():
    X = df.drop("label", axis=1)
    y = df["label"]
    
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette="coolwarm", alpha=0.6)
    plt.title("UMAP Projection (2D)")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend(title="Label")
    plt.savefig("reports/10_seq_umap_2d_projection.png")
    plt.close()

# ----------------------------
# Run all visualizations
# ----------------------------
if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)

    print("📊 Generating data visualizations...")

    plot_class_distribution()
    plot_top_syscalls(n=20)
    plot_correlation_heatmap()
    plot_pca_projection()
    plot_tsne_projection()
    plot_umap_projection()

    print("✅ Visualizations saved to 'reports/' folder.")