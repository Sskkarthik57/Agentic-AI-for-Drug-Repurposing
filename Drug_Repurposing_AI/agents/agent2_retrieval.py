"""
agents/agent2_retrieval.py — AGENT 2: Retrieval Agent
───────────────────────────────────────────────────────
Queries the FAISS index with fragments from known NDM-1 inhibitors.
Returns top-K similar fragments and maps them back to parent approved drugs.

Saves:
  outputs/retrieval_results.csv — candidate drugs with retrieval scores

Run: python agents/agent2_retrieval.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import faiss
import re
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose
from rdkit.Chem import AllChem, DataStructs
from transformers import AutoTokenizer, AutoModel
import torch

from config import (
    NDM1_BENCHMARK_CSV, FRAGMENT_META_CSV,
    RETRIEVAL_RESULTS_CSV, FAISS_INDEX_PATH,
    CHEMBERTA_MODEL, MORGAN_RADIUS, MORGAN_NBITS,
    MIN_FRAGMENT_ATOMS, MAX_FRAGMENT_ATOMS,
    EMBEDDING_TYPE, TOP_K, OUTPUT_DIR
)

OUTPUT_DIR.mkdir(exist_ok=True)


# ── Embed a single SMILES with Morgan FP ──────────────────────────────────────
def embed_morgan(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(MORGAN_NBITS, dtype=np.float32)
    fp  = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS
    )
    arr = np.zeros(MORGAN_NBITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ── Embed a single SMILES with ChemBERTa ─────────────────────────────────────
def load_chemberta():
    print(f"  Loading ChemBERTa: {CHEMBERTA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    model     = AutoModel.from_pretrained(CHEMBERTA_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    return tokenizer, model, device


def embed_chemberta(smi: str, tokenizer, model, device) -> np.ndarray:
    tokens = tokenizer(
        [smi], return_tensors="pt",
        padding=True, truncation=True, max_length=128
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        out = model(**tokens)
    cls = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return cls.astype(np.float32)


# ── Fragment a SMILES using BRICS ─────────────────────────────────────────────
def get_fragments(smi: str) -> list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    try:
        frags = BRICSDecompose(mol)
    except Exception:
        return []
    cleaned = []
    for f in frags:
        clean = re.sub(r'\[\d*\*\]', '[H]', f)
        fmol  = Chem.MolFromSmiles(clean)
        if fmol is None:
            continue
        n = fmol.GetNumAtoms()
        if MIN_FRAGMENT_ATOMS <= n <= MAX_FRAGMENT_ATOMS:
            cleaned.append(clean)
    return cleaned


# ── KNN retrieval ─────────────────────────────────────────────────────────────
def retrieve_candidates(query_smiles_list: list,
                        embedding_type: str = EMBEDDING_TYPE,
                        top_k: int = TOP_K) -> pd.DataFrame:
    """
    For each query SMILES (NDM-1 inhibitor fragment),
    find top-K similar fragments in the FAISS index.
    Map results back to parent approved drugs.

    Returns DataFrame with candidate drugs + scores.
    """
    # ── Load FAISS index ──────────────────────────────────────────────────────
    index_path = str(FAISS_INDEX_PATH).replace(
        ".faiss", f"_{embedding_type}.faiss"
    )
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            "Run agent1_fragment.py first."
        )
    index    = faiss.read_index(index_path)
    meta_df  = pd.read_csv(FRAGMENT_META_CSV, index_col='faiss_idx')
    print(f"  FAISS index loaded : {index.ntotal} vectors")
    print(f"  Metadata rows      : {len(meta_df)}")

    # ── Load embedding model ──────────────────────────────────────────────────
    if embedding_type == "chemberta":
        tokenizer, model, device = load_chemberta()

    # ── Query loop ────────────────────────────────────────────────────────────
    all_results = []

    print(f"\n  Querying {len(query_smiles_list)} NDM-1 inhibitor fragments...")

    for q_smi in tqdm(query_smiles_list, desc="  Retrieving"):
        # Embed query
        if embedding_type == "morgan":
            q_vec = embed_morgan(q_smi)
        else:
            q_vec = embed_chemberta(q_smi, tokenizer, model, device)

        # L2-normalize
        norm  = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        q_vec = q_vec.reshape(1, -1).astype(np.float32)

        # Search
        distances, indices = index.search(q_vec, top_k)

        for rank, (dist, idx) in enumerate(
            zip(distances[0], indices[0])
        ):
            if idx < 0 or idx >= len(meta_df):
                continue
            row = meta_df.iloc[idx]
            sim_score = float(1.0 / (1.0 + dist))   # convert L2 to similarity

            all_results.append({
                'query_fragment'   : q_smi,
                'matched_fragment' : row['fragment_clean'],
                'parent_chembl_id' : row['parent_chembl_id'],
                'parent_name'      : row.get('parent_name', ''),
                'parent_smiles'    : row['parent_smiles'],
                'rank'             : rank + 1,
                'l2_distance'      : float(dist),
                'similarity_score' : sim_score,
            })

    results_df = pd.DataFrame(all_results)

    # ── Aggregate: best similarity score per parent drug ──────────────────────
    best = (
        results_df.groupby('parent_chembl_id')
        .agg(
            parent_name    = ('parent_name', 'first'),
            parent_smiles  = ('parent_smiles', 'first'),
            best_sim_score = ('similarity_score', 'max'),
            n_hits         = ('query_fragment', 'count'),
            matched_frags  = ('matched_fragment',
                              lambda x: ' | '.join(x.unique()[:3]))
        )
        .reset_index()
        .sort_values('best_sim_score', ascending=False)
    )

    print(f"\n  Unique candidate drugs : {len(best)}")
    return best, results_df


# ── Main ──────────────────────────────────────────────────────────────────────
def run(embedding_type: str = EMBEDDING_TYPE):
    print("=" * 55)
    print("  AGENT 2 — RETRIEVAL AGENT")
    print("=" * 55)

    # ── Load NDM-1 benchmark — use TRAIN actives as queries ───────────────────
    bench = pd.read_csv(NDM1_BENCHMARK_CSV)
    train_actives = bench[
        (bench['split'] == 'train') & (bench['active'] == 1)
    ]['canonical_smiles'].tolist()

    print(f"\n  NDM-1 train actives : {len(train_actives)}")

    # ── Fragment NDM-1 inhibitors to get query fragments ─────────────────────
    print("  Extracting query fragments from NDM-1 inhibitors...")
    query_fragments = []
    for smi in tqdm(train_actives, desc="  Fragmenting queries"):
        frags = get_fragments(smi)
        query_fragments.extend(frags)

    # Deduplicate
    query_fragments = list(set(query_fragments))
    print(f"  Unique query fragments : {len(query_fragments)}")

    # Limit to top 200 to keep query time manageable
    if len(query_fragments) > 200:
        import random
        random.seed(42)
        query_fragments = random.sample(query_fragments, 200)
        print(f"  Sampled to 200 query fragments for efficiency")

    # ── Retrieve ──────────────────────────────────────────────────────────────
    candidates, raw_results = retrieve_candidates(
        query_smiles_list=query_fragments,
        embedding_type=embedding_type,
        top_k=TOP_K
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    candidates.to_csv(RETRIEVAL_RESULTS_CSV, index=False)
    raw_path = OUTPUT_DIR / "retrieval_raw.csv"
    raw_results.to_csv(raw_path, index=False)

    print(f"\n  Candidates saved  : {RETRIEVAL_RESULTS_CSV}")
    print(f"  Raw results saved : {raw_path}")
    print(f"\n  Top 5 candidates:")
    print(candidates[['parent_chembl_id', 'parent_name',
                       'best_sim_score', 'n_hits']].head())

    print("\n[✓] Agent 2 complete.\n")
    return candidates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default=EMBEDDING_TYPE,
                        choices=["morgan", "chemberta"])
    args = parser.parse_args()
    run(embedding_type=args.embedding)
