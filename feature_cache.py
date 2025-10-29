# feature_cache.py
import os, json
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback

def load_cache(cache_path):
    if not os.path.exists(cache_path): return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        return data["X"], data["paths"]
    except Exception:
        return None

def save_cache(cache_path, X, paths):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, X=X, paths=np.array(paths))

def build_with_cache(paths, feature_fn, cache_path=None):
    if cache_path:
        got = load_cache(cache_path)
        if got is not None:
            Xc, pc = got
            pc = [str(p) for p in pc]  # ensure list[str]
            if len(pc) == len(paths) and all(str(a)==str(b) for a,b in zip(pc, paths)):
                return Xc
    # compute
    X = feature_fn(paths)
    if cache_path:
        save_cache(cache_path, X, paths)
    return X

def iter_with_progress(seq, desc="processing"):
    return tqdm(seq, desc=desc)
