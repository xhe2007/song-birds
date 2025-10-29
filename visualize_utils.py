# visualize_utils.py
import os, io, math, csv, json
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------- Core scatter plots ----------
def save_pca_scatter(pc2, labels, out_path, title="PCA (2D) — colored by cluster"):
    plt.figure(figsize=(6,5))
    plt.scatter(pc2[:,0], pc2[:,1], c=labels, s=16, alpha=0.85)
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

def save_tsne_scatter(X, labels, out_path, seed=42, title="t-SNE (2D) — colored by cluster"):
    X = np.asarray(X, dtype=np.float64)
    ts = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=min(30, max(5, X.shape[0]//4)))
    Y = ts.fit_transform(X)
    plt.figure(figsize=(6,5))
    plt.scatter(Y[:,0], Y[:,1], c=labels, s=14, alpha=0.85)
    plt.title(title); plt.xlabel("tSNE-1"); plt.ylabel("tSNE-2"); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

# ---------- K selection curves ----------
def save_silhouette_curve(k_scores, out_path):
    # k_scores: list of (k, score)
    if not k_scores: return
    ks, ss = zip(*k_scores)
    plt.figure(figsize=(6,4))
    plt.plot(ks, ss, marker='o')
    plt.title("Silhouette vs K"); plt.xlabel("K"); plt.ylabel("Silhouette score"); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def save_elbow_curve(k_inertia, out_path):
    # k_inertia: list of (k, inertia)
    if not k_inertia: return
    ks, inert = zip(*k_inertia)
    plt.figure(figsize=(6,4))
    plt.plot(ks, inert, marker='o')
    plt.title("Elbow curve (Inertia vs K)"); plt.xlabel("K"); plt.ylabel("Inertia"); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# ---------- Cluster galleries ----------
def make_montage(image_paths, grid=None, thumb=128, pad=6, bg=(245,245,245)):
    if not image_paths: return None
    from PIL import Image
    if grid is None:
        cols = min(6, max(1, int(math.ceil(math.sqrt(len(image_paths))))))
        rows = int(math.ceil(len(image_paths)/cols))
    else:
        rows, cols = grid
    w = cols*thumb + (cols+1)*pad
    h = rows*thumb + (rows+1)*pad
    canvas = Image.new("RGB", (w, h), bg)
    idx = 0
    for p in image_paths[:rows*cols]:
        try:
            im = Image.open(p).convert("RGB")
            im = ImageOps.exif_transpose(im)
            im = ImageOps.fit(im, (thumb, thumb))
        except Exception:
            continue
        r, c = divmod(idx, cols)
        x = pad + c*(thumb+pad); y = pad + r*(thumb+pad)
        canvas.paste(im, (x, y)); idx += 1
    return canvas

def save_cluster_montages(cluster_to_paths, out_dir, thumb=128, pad=6):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for c, plist in sorted(cluster_to_paths.items()):
        m = make_montage(plist, grid=None, thumb=thumb, pad=pad)
        if m is None: continue
        path = os.path.join(out_dir, f"cluster_{c:02d}_montage.jpg")
        m.save(path, quality=90); saved.append(path)
    return saved

# ---------- Cluster exemplars & palette ----------
def argmin_cosine(a, b):
    # find index in b most similar to a using cosine distance
    # a: (d,), b: (n,d)
    a = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    sims = b_norm @ a
    return int(np.argmax(sims))

def save_cluster_exemplars(X, labels, paths, out_dir):
    """Pick 1 representative per cluster (closest to centroid in cosine space) and copy to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    reps = {}
    for c in sorted(set(labels)):
        idxs = np.where(labels == c)[0]
        Xc = X[idxs]
        centroid = Xc.mean(axis=0)
        j = argmin_cosine(centroid, Xc)
        reps[c] = paths[idxs[j]]
    # write an HTML index with thumbnails
    html = ["<html><body><h2>Cluster exemplars</h2><ul>"]
    for c in sorted(reps.keys()):
        p = reps[c]
        rel = os.path.relpath(p, out_dir)
        html.append(f'<li>Cluster {c}: <img src="{rel}" height="128"> {p}</li>')
    html.append("</ul></body></html>")
    with open(os.path.join(out_dir, "exemplars.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return reps

def save_cluster_hsv_histograms(paths, labels, out_dir, bins=32):
    os.makedirs(out_dir, exist_ok=True)
    from skimage.color import rgb2hsv
    def img_hist(p):
        try:
            im = Image.open(p).convert("RGB"); im = ImageOps.exif_transpose(im)
            arr = np.asarray(im, dtype=np.uint8)
        except Exception:
            return np.zeros((3, bins))
        hsv = rgb2hsv(arr/255.0)
        H = np.histogram(hsv[...,0], bins=bins, range=(0,1), density=True)[0]
        S = np.histogram(hsv[...,1], bins=bins, range=(0,1), density=True)[0]
        V = np.histogram(hsv[...,2], bins=bins, range=(0,1), density=True)[0]
        return np.stack([H,S,V], axis=0)

    for c in sorted(set(labels)):
        idx = np.where(labels==c)[0]
        if idx.size == 0: continue
        Hcum = np.zeros((3, bins))
        for i in idx:
            Hcum += img_hist(paths[i])
        Hcum /= max(1, idx.size)

        plt.figure(figsize=(7,4))
        x = np.arange(bins)
        plt.plot(x, Hcum[0], label="H")
        plt.plot(x, Hcum[1], label="S")
        plt.plot(x, Hcum[2], label="V")
        plt.title(f"Cluster {c}: mean HSV histogram")
        plt.xlabel("bin"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hsv_cluster_{c:02d}.png"), dpi=150)
        plt.close()

# ---------- Simple HTML gallery ----------
def save_html_gallery(paths, labels, out_path):
    cats = defaultdict(list)
    for p, c in zip(paths, labels):
        cats[int(c)].append(p)
    html = ["<html><body><h2>Cluster gallery</h2>"]
    for c in sorted(cats.keys()):
        html.append(f"<h3>Cluster {c}</h3><div>")
        for p in cats[c][:72]:
            html.append(f'<img src="{p}" height="128" style="margin:4px;">')
        html.append("</div>")
    html.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
