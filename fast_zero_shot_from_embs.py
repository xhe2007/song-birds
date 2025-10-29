import torch, numpy as np, open_clip

def _norm(x, eps=1e-8): return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def label_from_cached_embs(embs_np, kept_paths, species_names, scientific_map=None,
                           model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None, topk=3):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    tok = open_clip.get_tokenizer(model_name)

    # text centroids
    TEMPL = ["a painting of a {n} bird", "a depiction of a {n}"]
    txt_centroids = []
    for sp in species_names:
        alts = [sp] + ([scientific_map[sp]] if scientific_map and sp in scientific_map else [])
        prompts = [t.format(n=a) for a in alts for t in TEMPL]
        text = tok(prompts).to(device)
        tfeat = _norm(model.encode_text(text).float())
        txt_centroids.append(tfeat.mean(dim=0, keepdim=True))
    T = torch.cat(txt_centroids, dim=0)  # (S,D)

    I = torch.from_numpy(embs_np).to(device)       # (N,D) normalized already
    S = (I @ T.T).cpu().numpy()                    # cosine sims
    rows = []
    for i in range(S.shape[0]):
        idx = np.argsort(-S[i])[:topk]
        winners = [species_names[j] for j in idx]
        scores  = [float(S[i, j]) for j in idx]
        rows.append((kept_paths[i], winners, scores))
    return rows
