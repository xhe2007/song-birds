from visualize_utils import save_pca_scatter, save_tsne_scatter, save_silhouette_curve, save_elbow_curve, \
                             save_cluster_montages, save_cluster_exemplars, save_cluster_hsv_histograms, \
                             save_html_gallery
from feature_cache import build_with_cache
from species_labeling import load_prototypes, assign_species, save_species_csv

import os, glob, numpy as np, pandas as pd
from PIL import Image, ImageOps
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

IMG_DIR = "images"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Parameters
TARGET_SIZE = 256
HIST_BINS = 32
HOG_ORIENTATIONS = 9
HOG_PPC = (16,16)
HOG_CPB = (2,2)
K_MANUAL = None            # set an int to force K; otherwise silhouette search picks
K_RANGE = list(range(4, 13))

# Collect images
paths = []
for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG","*.JPEG"):
    paths.extend(glob.glob(os.path.join(IMG_DIR, ext)))
print(f"Found {len(paths)} images in '{IMG_DIR}'.")
if not paths:
    raise SystemExit("Put .jpg/.jpeg/.png images into 'images/' and re-run.")

def load_image(path, target=256):
    im = Image.open(path).convert("RGB")
    im = ImageOps.exif_transpose(im)  # respect orientation
    im = im.resize((target, target))
    return np.array(im)

def hsv_hist(img_rgb, bins=32):
    hsv = rgb2hsv(img_rgb / 255.0)
    h = np.histogram(hsv[...,0], bins=bins, range=(0,1), density=True)[0]
    s = np.histogram(hsv[...,1], bins=bins, range=(0,1), density=True)[0]
    v = np.histogram(hsv[...,2], bins=bins, range=(0,1), density=True)[0]
    return np.concatenate([h,s,v])

def hog_feat(img_rgb, orientations=9, ppc=(16,16), cpb=(2,2)):
    gray = rgb2gray(img_rgb)
    f = hog(gray, orientations=orientations,
            pixels_per_cell=ppc, cells_per_block=cpb,
            block_norm="L2-Hys", feature_vector=True)
    return f

def build_features(paths):
    feats = []
    for p in paths:
        img = load_image(p, target=TARGET_SIZE)
        feats.append(np.concatenate([
            hsv_hist(img, bins=HIST_BINS),
            hog_feat(img, orientations=HOG_ORIENTATIONS, ppc=HOG_PPC, cpb=HOG_CPB)
        ]))
    return np.vstack(feats)

# X = build_features(paths)
# print("Feature matrix:", X.shape)
def feature_fn(img_paths):
    feats = []
    for p in img_paths:
        img = load_image(p, target=TARGET_SIZE)
        feats.append(np.concatenate([
            hsv_hist(img, bins=HIST_BINS),
            hog_feat(img, orientations=HOG_ORIENTATIONS, ppc=HOG_PPC, cpb=HOG_CPB)
        ]))
    return np.vstack(feats)
cache_path = os.path.join(OUT_DIR, "features_cache.npz")
X = build_with_cache(paths, feature_fn, cache_path=cache_path)
print("Feature matrix:", X.shape)

#----
def pick_k_silhouette(X, k_list):
    best_k, best_score = None, -1
    for k in k_list:
        if X.shape[0] <= k:
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(X, labels)
        print(f"K={k:2d}  silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score



best_k, _ = pick_k_silhouette(X, K_RANGE) if K_MANUAL is None else (K_MANUAL, None)
K = best_k or 8
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
labels = km.labels_

#----
def pick_k_silhouette_with_elbow(X, k_list, seed=42):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    k_scores, k_inertia = [], []
    best_k, best_score = None, -1
    for k in k_list:
        if X.shape[0] <= k: continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        labels = km.labels_
        if len(set(labels)) <= 1: continue
        s = silhouette_score(X, labels)
        k_scores.append((k, float(s)))
        k_inertia.append((k, float(km.inertia_)))
        if s > best_score: best_k, best_score = k, s
    return best_k, best_score, k_scores, k_inertia

best_k, best_score, k_scores, k_inertia = pick_k_silhouette_with_elbow(X, K_RANGE, seed=42)
K = best_k or 8
print("Picked K:", K, "silhouette:", best_score)
save_silhouette_curve(k_scores, os.path.join(OUT_DIR, "silhouette_curve.png"))
save_elbow_curve(k_inertia, os.path.join(OUT_DIR, "elbow_curve.png"))

# ---- PCA (keep what you had) ----
X = np.ascontiguousarray(X, dtype=np.float64)
if not np.isfinite(X).all():
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

pc = PCA(n_components=2, random_state=42).fit_transform(X)   # shape (n, 2)
labels = np.asarray(labels).ravel()                          # shape (n,)
paths_list = [os.fspath(p) for p in paths]                   # force plain str paths

n = len(paths_list)
assert pc.shape == (n, 2)
assert labels.shape == (n,)

# --- inputs assumed already computed above ---
# paths (list of str or Path), labels (array-like shape (n,)), pc (n,2)
# OUT_DIR exists

import os, csv
import numpy as np
from collections import defaultdict

# 1) Normalize types to plain 1-D scalars
paths_str  = [os.fspath(p) for p in paths]
labels_1d  = np.asarray(labels, dtype=np.int32).ravel()
pc1        = np.asarray(pc[:, 0], dtype=np.float64).ravel()
pc2        = np.asarray(pc[:, 1], dtype=np.float64).ravel()

n = len(paths_str)
assert labels_1d.shape == (n,), (labels_1d.shape, n)
assert pc1.shape == (n,) and pc2.shape == (n,), (pc1.shape, pc2.shape, n)

# 2) Write clusters.csv manually (no pandas)
clusters_csv = os.path.join(OUT_DIR, "clusters.csv")
with open(clusters_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "cluster", "pc1", "pc2"])
    for i in range(n):
        w.writerow([paths_str[i], int(labels_1d[i]), float(pc1[i]), float(pc2[i])])
print("Saved:", clusters_csv)


size_tbl = df.groupby("cluster")["path"].count().rename("count").reset_index().sort_values("count", ascending=False)
size_tbl.to_csv(os.path.join(OUT_DIR, "cluster_sizes.csv"), index=False)
print("Saved: outputs/cluster_sizes.csv")
print(size_tbl.head(10))


# Optional montages (quick visual check)
def make_montage(paths, grid=(4,4), thumb=128, pad=6):
    n = grid[0]*grid[1]
    sel = paths[:n]
    w = grid[1]*thumb + (grid[1]+1)*pad
    h = grid[0]*thumb + (grid[0]+1)*pad
    canvas = Image.new("RGB", (w, h), (245,245,245))
    for idx, p in enumerate(sel):
        im = Image.open(p).convert("RGB")
        im = ImageOps.exif_transpose(im)
        im = ImageOps.fit(im, (thumb, thumb))
        r = idx // grid[1]; c = idx % grid[1]
        x = pad + c*(thumb+pad); y = pad + r*(thumb+pad)
        canvas.paste(im, (x, y))
    return canvas

for c, g in df.groupby("cluster"):
    m = make_montage(g["path"].tolist(), grid=(4,4), thumb=128, pad=6)
    out_path = os.path.join(OUT_DIR, f"cluster_{c:02d}_montage.jpg")
    m.save(out_path, quality=90)
    print("Saved montage:", out_path)

# 3) Compute cluster sizes without pandas
unique, counts = np.unique(labels_1d, return_counts=True)
sizes = list(zip(unique.tolist(), counts.tolist()))

sizes_csv = os.path.join(OUT_DIR, "cluster_sizes.csv")
with open(sizes_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cluster", "count"])
    for c, cnt in sizes:
        w.writerow([int(c), int(cnt)])
print("Saved:", sizes_csv)

# 4) Build cluster -> paths map for montages (instead of df.groupby)
cluster_to_paths = defaultdict(list)
for p, c in zip(paths_str, labels_1d):
    cluster_to_paths[int(c)].append(p)

from PIL import Image, ImageOps
import numpy as np

def make_montage(image_paths, grid=None, thumb=128, pad=6, bg=(245, 245, 245)):
    """
    Create a simple montage from a list of image paths.

    Params:
      image_paths : list[str]
      grid        : (rows, cols) or None to auto-layout
      thumb       : thumbnail size (square)
      pad         : padding (px)
      bg          : background RGB tuple
    """
    if not image_paths:
        raise ValueError("No images provided to make_montage().")

    # Auto grid if not supplied
    if grid is None:
        cols = min(4, max(1, int(np.ceil(np.sqrt(len(image_paths))))))
        rows = int(np.ceil(len(image_paths) / cols))
    else:
        rows, cols = grid

    # Canvas size
    w = cols * thumb + (cols + 1) * pad
    h = rows * thumb + (rows + 1) * pad
    canvas = Image.new("RGB", (w, h), bg)

    # Paste thumbnails
    placed = 0
    for idx, p in enumerate(image_paths[: rows * cols]):
        try:
            im = Image.open(p).convert("RGB")
            im = ImageOps.exif_transpose(im)
            im = ImageOps.fit(im, (thumb, thumb))
        except Exception as e:
            # Skip unreadable/corrupt files but keep layout stable
            continue
        r = idx // cols
        c = idx % cols
        x = pad + c * (thumb + pad)
        y = pad + r * (thumb + pad)
        canvas.paste(im, (x, y))
        placed += 1

    if placed == 0:
        raise ValueError("All images failed to load in make_montage().")
    return canvas

# 5) Montages (unchanged except iterating over our dict)
for c, plist in cluster_to_paths.items():
    m = make_montage(plist, grid=(4,4), thumb=128, pad=6)
    out_path = os.path.join(OUT_DIR, f"cluster_{c:02d}_montage.jpg")
    m.save(out_path, quality=90)
    print("Saved montage:", out_path)



# (already have pc ∈ R^{n×2}, labels, paths_str)
save_pca_scatter(pc, labels, os.path.join(OUT_DIR, "pca_scatter.png"))
save_tsne_scatter(X, labels, os.path.join(OUT_DIR, "tsne_scatter.png"))
save_html_gallery(paths_str, labels, os.path.join(OUT_DIR, "cluster_gallery.html"))

# Cluster montages
from collections import defaultdict
cluster_to_paths = defaultdict(list)
for p, c in zip(paths_str, labels):
    cluster_to_paths[int(c)].append(p)
save_cluster_montages(cluster_to_paths, os.path.join(OUT_DIR, "montages"))
save_cluster_hsv_histograms(paths_str, labels, os.path.join(OUT_DIR, "cluster_hsv"))
save_cluster_exemplars(X, labels, paths_str, os.path.join(OUT_DIR, "exemplars"))

# Example species list aligned with motifs in Song bird-and-flower paintings
species_common = [
    "crane", "duck", "sparrow", "wagtail", "magpie",
    "pheasant", "mandarin duck", "peafowl", "goshawk"
]
# Optional scientific names to strengthen prompts (sample mapping)
scientific = {
    "mandarin duck": "Aix galericulata",
    "wagtail": "Motacilla alba",
    "magpie": "Pica pica",
    "pheasant": "Phasianus colchicus",
    "peafowl": "Pavo muticus",
    "goshawk": "Accipiter gentilis"
}

from zero_shot_labeling import zero_shot_label, save_zero_shot_csv
rows, _ = zero_shot_label(paths_str, species_common, scientific_map=scientific,
                          model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", topk=3)
save_zero_shot_csv(rows, os.path.join(OUT_DIR, "species_labels_zero_shot.csv"), topk=3)
