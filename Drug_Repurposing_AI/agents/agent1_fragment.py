"""
agents/agent1_fragment.py — AGENT 1: Fragment Agent
─────────────────────────────────────────────────────
Step 1: BRICS decompose all approved drugs (with timeout protection)
Step 2: Embed fragments (Morgan FP baseline + ChemBERTa learned)
Step 3: Build FAISS index for fast KNN retrieval

Run: python agents/agent1_fragment.py --embedding morgan
     python agents/agent1_fragment.py --embedding chemberta
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import re
import faiss
import threading
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem.BRICS import BRICSDecompose
from rdkit.Chem import AllChem, DataStructs
from transformers import AutoTokenizer, AutoModel
import torch

from config import (
    APPROVED_DRUGS_CSV, FRAGMENTS_CSV,
    FAISS_INDEX_PATH, FRAGMENT_META_CSV,
    MORGAN_RADIUS, MORGAN_NBITS,
    CHEMBERTA_MODEL,
    MIN_FRAGMENT_ATOMS, MAX_FRAGMENT_ATOMS,
    EMBEDDING_TYPE, OUTPUT_DIR
)

OUTPUT_DIR.mkdir(exist_ok=True)
RDLogger.DisableLog('rdApp.*')   # suppress RDKit warnings


# ── Timeout-safe BRICS ────────────────────────────────────────────────────────
def brics_safe(mol, timeout_sec=3):
    """
    Run BRICSDecompose in a thread. Returns None if it times out.
    Works on Windows (no SIGALRM available there).
    """
    result = [None]

    def _run():
        try:
            result[0] = list(BRICSDecompose(mol))
        except Exception:
            result[0] = []

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        return None   # timed out
    return result[0]


# ── STEP 1: Fragmentation ─────────────────────────────────────────────────────
def fragment_drugs(drugs_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[1] BRICS Fragmentation (3s timeout per molecule)...")
    records  = []
    failed   = 0
    timeouts = 0

    for _, row in tqdm(drugs_df.iterrows(), total=len(drugs_df),
                       desc="  Fragmenting"):
        smi = str(row['smiles'])

        # Skip very large SMILES upfront — always cause timeouts
        if len(smi) > 400:
            timeouts += 1
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue

        # Skip molecules with too many heavy atoms
        if mol.GetNumHeavyAtoms() > 70:
            timeouts += 1
            continue

        frags = brics_safe(mol, timeout_sec=3)

        if frags is None:
            # Timed out — use whole molecule as single "fragment"
            timeouts += 1
            try:
                frags = [Chem.MolToSmiles(mol)]
            except Exception:
                continue

        for frag_smi in frags:
            clean_smi = re.sub(r'\[\d*\*\]', '[H]', frag_smi)
            frag_mol  = Chem.MolFromSmiles(clean_smi)
            if frag_mol is None:
                continue
            n = frag_mol.GetNumHeavyAtoms()
            if n < MIN_FRAGMENT_ATOMS or n > MAX_FRAGMENT_ATOMS:
                continue

            records.append({
                'parent_chembl_id': row['chembl_id'],
                'parent_name'     : str(row.get('name', '')),
                'parent_smiles'   : smi,
                'fragment_smiles' : frag_smi,
                'fragment_clean'  : clean_smi,
                'n_atoms'         : n,
            })

    if not records:
        raise RuntimeError("No fragments generated — check SMILES in approved_drugs_chembl.csv")

    df        = pd.DataFrame(records)
    df        = df.drop_duplicates(subset=['parent_chembl_id', 'fragment_clean'])
    processed = len(drugs_df) - failed - timeouts

    print(f"  Drugs processed  : {processed}")
    print(f"  Timeouts/skipped : {timeouts}")
    print(f"  Invalid SMILES   : {failed}")
    print(f"  Total fragments  : {len(df)}")
    print(f"  Unique frag SMILES: {df['fragment_clean'].nunique()}")
    if processed > 0:
        print(f"  Avg frags/drug   : {len(df)/processed:.1f}")
    return df


# ── STEP 2a: Morgan FP ────────────────────────────────────────────────────────
def morgan_embed(smiles_list: list) -> np.ndarray:
    print(f"  Morgan FP: {MORGAN_NBITS}-bit radius={MORGAN_RADIUS}")
    gen        = AllChem.GetMorganGenerator(radius=MORGAN_RADIUS,
                                            fpSize=MORGAN_NBITS)
    embeddings = []
    for smi in tqdm(smiles_list, desc="  Morgan FP"):
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(MORGAN_NBITS, dtype=np.float32)
        if mol:
            fp = gen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr)
        embeddings.append(arr)
    return np.array(embeddings, dtype=np.float32)


# ── STEP 2b: ChemBERTa ────────────────────────────────────────────────────────
def chemberta_embed(smiles_list: list, batch_size: int = 32) -> np.ndarray:
    print(f"  Loading ChemBERTa: {CHEMBERTA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    model     = AutoModel.from_pretrained(CHEMBERTA_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"  Device: {device}")

    all_emb = []
    for i in tqdm(range(0, len(smiles_list), batch_size),
                  desc="  ChemBERTa batches"):
        batch  = smiles_list[i: i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt",
                           padding=True, truncation=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
        all_emb.append(out.last_hidden_state[:, 0, :].cpu().numpy())

    return np.vstack(all_emb).astype(np.float32)


# ── STEP 3: FAISS Index ───────────────────────────────────────────────────────
def build_faiss_index(embeddings, index_path, meta_path, fragment_df):
    print(f"\n[3] Building FAISS index...")
    print(f"  Shape: {embeddings.shape}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_n = (embeddings / norms).astype(np.float32)

    index = faiss.IndexFlatL2(emb_n.shape[1])
    index.add(emb_n)
    print(f"  Vectors: {index.ntotal}")

    faiss.write_index(index, str(index_path))
    fragment_df.reset_index(drop=True).to_csv(
        meta_path, index=True, index_label='faiss_idx'
    )
    print(f"  Saved: {index_path}")
    print(f"  Meta : {meta_path}")
    return index


# ── Main ──────────────────────────────────────────────────────────────────────
def run(embedding_type: str = EMBEDDING_TYPE):
    print("=" * 55)
    print("  AGENT 1 — FRAGMENT AGENT")
    print("=" * 55)

    drugs_df = pd.read_csv(APPROVED_DRUGS_CSV)
    print(f"\n  Approved drugs loaded: {len(drugs_df)}")

    frag_df = fragment_drugs(drugs_df)
    frag_df.to_csv(FRAGMENTS_CSV, index=False)
    print(f"  Fragments CSV saved: {FRAGMENTS_CSV}")

    smiles_list = frag_df['fragment_clean'].tolist()
    print(f"\n  Fragments to embed: {len(smiles_list)}")

    print(f"\n[2] Embedding ({embedding_type.upper()})...")
    embeddings = morgan_embed(smiles_list) if embedding_type == "morgan" \
                 else chemberta_embed(smiles_list)

    emb_path = OUTPUT_DIR / f"embeddings_{embedding_type}.npy"
    np.save(str(emb_path), embeddings)
    print(f"  Embeddings saved: {emb_path}  shape={embeddings.shape}")

    index_path = str(FAISS_INDEX_PATH).replace(".faiss",
                                               f"_{embedding_type}.faiss")
    build_faiss_index(embeddings, index_path,
                      str(FRAGMENT_META_CSV), frag_df)

    print("\n[✓] Agent 1 complete.\n")
    return frag_df, embeddings


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default=EMBEDDING_TYPE,
                        choices=["morgan", "chemberta"])
    args = parser.parse_args()
    run(embedding_type=args.embedding)