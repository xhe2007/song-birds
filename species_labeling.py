# species_labeling.py
import os, glob, csv
import numpy as np
from collections import defaultdict

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T  # (n, m)

def load_prototypes(proto_dir, feature_fn):
    if not os.path.isdir(proto_dir): return {}
    species = {}
    for name in sorted(os.listdir(proto_dir)):
        folder = os.path.join(proto_dir, name)
        if not os.path.isdir(folder): continue
        imgs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG","*.JPEG"):
            imgs += glob.glob(os.path.join(folder, ext))
        if not imgs: continue
        # feature average per species
        F = feature_fn(imgs)  # (k, d)
        species[name] = F.mean(axis=0)
    return species  # dict: species -> (d,)

def assign_species(X, paths, species_centroids, topk=1):
    if not species_centroids: return None
    names = sorted(species_centroids.keys())
    C = np.stack([species_centroids[n] for n in names], axis=0)  # (m, d)
    sims = cosine_similarity(X, C)  # (n, m)
    idx = np.argsort(-sims, axis=1)[:, :topk]
    conf = np.take_along_axis(sims, idx, axis=1)
    # build rows
    rows = []
    for i, p in enumerate(paths):
        winners = [names[j] for j in idx[i]]
        scores = [float(c) for c in conf[i]]
        rows.append((p, winners[0], scores[0]) if topk==1 else (p, winners, scores))
    return rows, names

def save_species_csv(rows, out_csv, topk=1):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if topk == 1:
            w.writerow(["path", "pred_species", "confidence"])
            for p, sp, s in rows:
                w.writerow([p, sp, s])
        else:
            w.writerow(["path", "topk_species", "topk_scores"])
            for p, sps, ss in rows:
                w.writerow([p, "|".join(sps), "|".join(map(lambda x: f"{x:.4f}", ss))])
                                