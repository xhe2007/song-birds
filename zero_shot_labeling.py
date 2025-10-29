# zero_shot_labeling.py
import os, csv, re, torch
import numpy as np
from PIL import Image
import open_clip

# --- Helper: basic CLIP preprocessing ---
def _preprocess(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer

def _norm(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# --- Build class prompts (both common & scientific names help) ---
def build_prompts(species_names, scientific_map=None, templates=None):
    if templates is None:
        templates = [
            "a painting of a {name} bird",
            "a depiction of a {name}",
            "a Song-dynasty painting of a {name}",
            "a Chinese ink painting of a {name}",
            "a photo of a {name} bird"
        ]
    prompts = []
    for sp in species_names:
        alts = [sp]
        if scientific_map and sp in scientific_map:
            alts.append(scientific_map[sp])  # e.g., "Aix galericulata"
        # dedupe and clean
        cand = list(dict.fromkeys(alts))
        # expand with templates
        for nm in cand:
            for t in templates:
                prompts.append(t.format(name=nm))
    return prompts

# --- Encode class prompts into a single centroid per species ---
def encode_class_text_centroids(model, tokenizer, device, species_names, scientific_map=None):
    species_to_text = {}
    for sp in species_names:
        prompts = build_prompts([sp], scientific_map=scientific_map)
        with torch.no_grad():
            tok = tokenizer(prompts)
            text = tok.to(device)
            tfeat = model.encode_text(text)
            tfeat = _norm(tfeat.float())
            centroid = tfeat.mean(dim=0, keepdim=True)  # (1, D)
        species_to_text[sp] = centroid
    return species_to_text

# --- Encode one image path ---
def encode_image(model, preprocess, device, path):
    with torch.no_grad():
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        feat = model.encode_image(img)
        feat = _norm(feat.float())  # (1, D)
        return feat

# --- Predict top-k species for each image ---
def zero_shot_label(paths, species_names, scientific_map=None,
                    model_name="ViT-B-32", pretrained="laion2b_s34b_b79k",
                    device=None, topk=3):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = _preprocess(model_name, pretrained, device)
    # text centroids
    sp2txt = encode_class_text_centroids(model, tokenizer, device, species_names, scientific_map)
    names = list(sp2txt.keys())
    T = torch.cat([sp2txt[n] for n in names], dim=0)  # (S, D)

    rows = []
    with torch.no_grad():
        for p in paths:
            I = encode_image(model, preprocess, device, p)  # (1, D)
            sim = (I @ T.T).squeeze(0)  # cosine since normalized
            k = min(topk, sim.numel())
            scores, idx = torch.topk(sim, k)
            winners = [names[i] for i in idx.tolist()]
            rows.append((p, winners, [float(s) for s in scores.tolist()]))
    return rows, names

def save_zero_shot_csv(rows, out_csv, topk=3):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "topk_species", "topk_scores"])
        for p, sps, ss in rows:
            w.writerow([p, "|".join(sps[:topk]), "|".join(f"{x:.4f}" for x in ss[:topk])])
