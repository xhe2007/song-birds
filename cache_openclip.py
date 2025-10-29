import os, torch, numpy as np
from PIL import Image
import open_clip

def _norm(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def encode_images_openclip(paths, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k",
                           device=None, batch_size=32, local_ckpt=None):
    device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=None if local_ckpt else pretrained, device=device
    )
    if local_ckpt:
        sd = torch.load(local_ckpt, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    model.eval()

    # encode batch-by-batch
    feats, keep = [], []
    batch, keep_p = [], []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        batch.append(preprocess(img)); keep_p.append(p)
        if len(batch) == batch_size:
            I = torch.stack(batch).to(device)
            E = _norm(model.encode_image(I).float()).cpu().numpy()
            feats.append(E); keep.extend(keep_p)
            batch, keep_p = [], []
    if batch:
        I = torch.stack(batch).to(device)
        E = _norm(model.encode_image(I).float()).cpu().numpy()
        feats.append(E); keep.extend(keep_p)

    if not feats: return np.empty((0,)), []
    return np.vstack(feats), keep

def save_embeddings_npz(out_path, embs, paths):
    np.savez_compressed(out_path, embs=embs, paths=np.array(paths))

def load_embeddings_npz(path):
    if not os.path.exists(path): return None, None
    d = np.load(path, allow_pickle=False)
    return d["embs"], d["paths"].tolist()
