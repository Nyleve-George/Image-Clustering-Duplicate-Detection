# Image Clustering and Duplicate Detection Using InceptionV3 + K-Means

> COMP 603 Assignment | 2026  
> Dataset: 304 social media images | Embeddings: InceptionV3 (2048-dimensional)

---

## Project Overview

A company collects images from competitors' social media pages. This project groups visually similar images together using unsupervised clustering techniques, organises them into separate folders, and identifies duplicate images — all without any image labels or manual tagging.

Images are represented as **2048-dimensional embedding vectors** extracted using the **InceptionV3** pretrained CNN model.

---

## Tasks Completed

| Task | Description | Method | Status |
|---|---|---|---|
| (a) Cluster Analysis | Find optimal k and cluster images | K-Means + Agglomerative (elbow + silhouette) | ✅ Complete |
| (b) Cluster Dictionary | Map each cluster label to image filenames | Python dict from K-Means labels | ✅ Complete |
| (c) Organise & Display | Move images into subfolders, show 5 samples per cluster | `shutil.move` + matplotlib gallery | ✅ Complete |
| (d) Duplicate Detection | Find near-identical image pairs | Pairwise cosine distance on L2-normalised embeddings | ✅ Complete |

---

## Results

### Optimal Clusters: K = 5 (Silhouette = 0.1385)

| Cluster | Size | Visual Theme |
|---|---|---|
| 0 | 40 images | Birds of prey — eagles, hawks, large raptors |
| 1 | 90 images | Small & medium birds — songbirds, parrots, perched birds |
| 2 | 49 images | Cats — domestic cats, indoor and outdoor |
| 3 | 110 images | Dogs & cats mixed — small breeds, indoor pets |
| 4 | 15 images | Dogs — predominantly small breeds (poodles, toy dogs) |

### Model Comparison

| Method | Best k | Best Silhouette Score |
|---|---|---|
| **K-Means** | **5** | **0.1385** |
| Agglomerative (Ward) | 10 | 0.1484 |

K-Means was selected as the final algorithm — its silhouette score difference from Agglomerative is under 0.01, while being significantly faster and more scalable.

### Duplicate Detection
- **3 exact duplicate pairs** found at cosine distance threshold < 0.01
- All 3 pairs had cosine distance = 0.0 (pixel-identical images)

---

## Repository Structure

```
├── Task1_Image_Clustering(V2).ipynb       # Main analysis notebook
├── Task1_Image_Clustering(V2).pdf         # Exported notebook with outputs
├── image_vectors_InceptionV3.csv          # 2048-dim InceptionV3 embeddings
├── collected_images/                      # Source images (304 JPGs)
├── clustered_images/                      # Output: images sorted by cluster
│   ├── cluster_0/                         # 40 images — birds of prey
│   ├── cluster_1/                         # 90 images — small birds
│   ├── cluster_2/                         # 49 images — cats
│   ├── cluster_3/                         # 110 images — dogs & cats
│   └── cluster_4/                         # 15 images — small dogs
└── README.md
```

---

## Methodology

### 1. Preprocessing
- **L2 Normalisation** — applied to all 2048-dim vectors so Euclidean and cosine distance become equivalent, aligning K-Means with the correct similarity measure for neural embeddings
- **PCA (50 components)** — reduces dimensionality for faster clustering while retaining 64.38% of variance; full 2048-dim vectors preserved for duplicate detection

### 2. Cluster Optimisation
- **Elbow Method** — tested k = 2 to 15, identifies where inertia decrease slows
- **Silhouette Score** — quantifies how well each point fits its assigned cluster vs. nearest alternative cluster (range: -1 to +1)

### 3. Clustering Algorithms
- **K-Means** — fast, scalable, random initialisation with `n_init=10`
- **Agglomerative (Ward linkage)** — deterministic, bottom-up hierarchical, minimises within-cluster variance at each merge

### 4. Duplicate Detection
- Pairwise **cosine distances** computed on full L2-normalised 2048-dim vectors
- Threshold: cosine distance < 0.01
- 3 duplicate pairs detected, all with distance = 0.0

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data loading and matrix operations |
| Scikit-learn | KMeans, AgglomerativeClustering, PCA, silhouette scoring |
| Matplotlib | Cluster galleries and evaluation plots |
| shutil / os | File organisation into cluster folders |
| Jupyter Notebook | Analysis and reporting |

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
```

### 2. Ensure folder structure
Place all images in a folder named `collected_images/` and the CSV file in the project root.

### 3. Run the notebook
```bash
jupyter notebook "Task1_Image_Clustering(V2).ipynb"
```

---

## Key Design Decisions

- **PCA before clustering** — speeds up K-Means significantly; full vectors still used for duplicate detection where precision matters
- **L2 normalisation** — InceptionV3 embeddings vary in magnitude independently of visual content; normalisation corrects for this
- **K-Means over Agglomerative** — near-identical silhouette scores (<0.01 difference) but K-Means is faster and scales to larger image collections
- **Silhouette scores below 0.20** are typical for real-world social media embeddings — semantic coherence in cluster galleries confirms meaningful groupings despite modest mathematical separation
