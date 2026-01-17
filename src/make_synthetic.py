# src/make_synthetic.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    # ===== settings =====
    seed = 42
    n = 500          # samples
    d = 5            # observed dims
    k = 2            # latent dims
    sigma = 0.3      # noise std

    rng = np.random.default_rng(seed)

    # ===== latent factors Z: (n, k) =====
    Z = rng.normal(loc=0.0, scale=1.0, size=(n, k))

    # ===== mixing matrix W: (k, d) =====
    W = np.array([
        [1.2, 1.0, 0.8, 0.1, 0.0],
        [0.0, 0.2, 0.3, 1.0, 1.1],
    ], dtype=float)

    # sanity
    assert W.shape == (k, d)

    # ===== noise E: (n, d) =====
    E = rng.normal(loc=0.0, scale=sigma, size=(n, d))

    # ===== observed X: (n, d) =====
    X = Z @ W + E

    # ===== save =====
    data_dir = Path("../data")
    data_dir.mkdir(parents=True, exist_ok=True)

    cols = [f"X{i}" for i in range(1, d + 1)]
    df = pd.DataFrame(X, columns=cols)
    df.to_csv(data_dir / "X.csv", index=False)

    meta = {
        "seed": seed,
        "n": n,
        "d": d,
        "k_latent": k,
        "sigma": sigma,
        "W": W.tolist(),
        "columns": cols,
    }
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:", data_dir / "X.csv", "and", data_dir / "meta.json")


if __name__ == "__main__":
    main()
