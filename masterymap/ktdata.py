"""
Public tutoring datasets for the benchmark, as preprocessed and published by Gervet, Koedinger, Schneider & Mitchell
(2020) "When is deep learning the best approach to knowledge tracing?" (Journal of Educational Data Mining 12(3)),
https://github.com/theophilee/learner-performance-prediction. Downloaded on first use from a pinned commit and
checked by SHA-256; kept in data_cache/kt/ (not part of this repository).

Each dataset: one row per answer (user_id, item_id, timestamp, correct, skill_id) in time order within each learner,
and q_mat.npz, the item-by-skill matrix (scipy CSR arrays, read here with allow_pickle=False).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = os.environ.get("MM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data_cache", "kt")
COMMIT = "a7ae193aa6957003a764aed7c95b07666fd4f1da"
URL = "https://raw.githubusercontent.com/theophilee/learner-performance-prediction/{commit}/data/{name}/{file}"
SHA256 = {
    "assistments09": ("40d0c4b01f99910c18cd8728f3bea7366deb3a9113bb77271daecb00b2da9669",
                      "3175e1be67a41da872cc8ee75c175a35073fb6b71f0da1585ba7f71192aa6b06"),
    "assistments15": ("612cecbf622bfcc2bc386720e77292f99df362512e1c35e21ea0e2311a2db656",
                      "8f48ff106a052bc4bd3361a9b0636aeb50c317a3082c2c469a3ef66624a619ce"),
    "assistments17": ("936fee58c1988084eb1633dfcf91e89d025cafb68ae2112b9d145a9f8f10e2e5",
                      "0c489eca588c1c768105e08002431f1803ad0e1edb4bdd4f9bf9748a1ae862f0"),
    "statics": ("6633163dc95f3d13fb39efe4e366c905a3b7f636085e46fe0c815a9f9fdd9d03",
                "03fc17339aa4f0b56963bb810143fdf281b313e467a174c067e927314fba1558"),
    "algebra05": ("e38d83265ed77bb751313a3fb0183c9c8d793bba3873113bc0cf0bf74d4ccb62",
                  "c6cd832b9b5e9db76e993816b5db4f8aad51d219de61615bb87e93ed2bc25e52"),
    "spanish": ("14932238cf7078859343fdafb52a987316e71429e4ea7c83dfa681e7c50877b5",
                "565f00f74b230e2827c2e9dc0020fb7c3ee0ef43908423bb9ed497d32274766d"),
}
FILES = ("preprocessed_data.csv", "q_mat.npz")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name):
    """Paths of the dataset's two files, downloading them if needed; refuses files with the wrong checksum."""
    out = []
    for file, want in zip(FILES, SHA256[name]):
        path = os.path.join(CACHE, name, file)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".part"
            urllib.request.urlretrieve(URL.format(commit=COMMIT, name=name, file=file), tmp)
            os.replace(tmp, path)
        got = _sha(path)
        if got != want:
            raise ValueError(f"{name}/{file}: checksum {got[:12]}... is not the published {want[:12]}...; delete "
                             f"{path} and run again")
        out.append(path)
    return out


def load(name):
    """(answers DataFrame, Q sparse matrix) of a benchmark dataset."""
    csv, qm = fetch(name)
    df = pd.read_csv(csv, sep="\t", usecols=["user_id", "item_id", "timestamp", "correct", "skill_id"])
    z = np.load(qm, allow_pickle=False)
    if bytes(z["format"]) != b"csr":
        raise ValueError("unexpected q-matrix format")
    Q = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(int(v) for v in z["shape"]))
    return df, Q
