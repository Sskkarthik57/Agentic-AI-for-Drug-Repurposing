"""
agents/agent3_scoring.py — AGENT 3: Scoring Agent
───────────────────────────────────────────────────
Scores retrieved drug candidates against NDM-1 using:
  1. DeepPurpose DTI model — predicted binding affinity
  2. Lipinski Rule-of-5 — drug-likeness filter
  3. Combined score — rank final candidates

Saves:
  outputs/scored_candidates.csv
  outputs/final_ranked_drugs.csv   ← main output

Run: python agents/agent3_scoring.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

from config import (
    RETRIEVAL_RESULTS_CSV, SCORED_CANDIDATES_CSV,
    FINAL_RANKED_CSV, NDM1_SEQUENCE,
    LIPINSKI_MW_MAX, LIPINSKI_LOGP_MAX,
    LIPINSKI_HBD_MAX, LIPINSKI_HBA_MAX,
    OUTPUT_DIR
)

OUTPUT_DIR.mkdir(exist_ok=True)


# ── Lipinski Rule-of-5 ────────────────────────────────────────────────────────
def lipinski_filter(smiles: str) -> dict:
    """
    Returns dict with RDKit-computed drug-likeness properties.
    pass_ro5 = True if molecule satisfies Lipinski's Rule of Five.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {'mw': None, 'logp': None, 'hbd': None,
                'hba': None, 'pass_ro5': False}

    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)

    pass_ro5 = (
        mw   <= LIPINSKI_MW_MAX  and
        logp <= LIPINSKI_LOGP_MAX and
        hbd  <= LIPINSKI_HBD_MAX  and
        hba  <= LIPINSKI_HBA_MAX
    )

    return {'mw': round(mw, 2), 'logp': round(logp, 2),
            'hbd': hbd, 'hba': hba, 'pass_ro5': pass_ro5}


# ── DeepPurpose DTI Scoring ───────────────────────────────────────────────────
def dti_score_deepurpose(smiles_list: list,
                          target_seq: str = NDM1_SEQUENCE) -> list:
    """
    Predict drug-target interaction scores using DeepPurpose.
    Returns list of predicted binding scores (higher = stronger binding).
    Falls back to similarity-based proxy if DeepPurpose fails.
    """
    try:
        from DeepPurpose import utils as dp_utils
        from DeepPurpose import CompoundPred

        print("  Loading DeepPurpose model (BindingDB_Kd)...")
        model = CompoundPred.MPNN_CNN(
            'BindingDB_Kd',
            pretrained=True,
            device="cpu"
        )

        scores = []
        for smi in tqdm(smiles_list, desc="  DTI scoring"):
            try:
                result = model.predict([smi], [target_seq])
                # Lower Kd = stronger binding → invert for ranking
                kd_val = float(result[0]) if result else 999.0
                # Convert: score = -log10(Kd) — higher is better
                score  = -np.log10(max(kd_val, 1e-6))
                scores.append(round(score, 4))
            except Exception:
                scores.append(0.0)
        return scores

    except ImportError:
        print("  [!] DeepPurpose not installed — using Morgan similarity proxy")
        return _morgan_similarity_proxy(smiles_list)

    except Exception as e:
        print(f"  [!] DeepPurpose error ({e}) — using proxy scoring")
        return _morgan_similarity_proxy(smiles_list)


def _morgan_similarity_proxy(smiles_list: list) -> list:
    """
    Fallback: use average Morgan Tanimoto similarity to known
    NDM-1 inhibitors as a proxy for binding score.
    """
    from rdkit.Chem import AllChem, DataStructs

    # A few known NDM-1 inhibitor SMILES as reference
    known_ndm1_smiles = [
        "O=C(O)CN1CCN(CC(=O)O)CCN(CC(=O)O)CC1",   # EDTA-like chelator
        "O=C(O)c1ccc(cc1)S(=O)(=O)N",               # sulfonamide scaffold
        "OC(=O)c1ccc(cc1)C(=O)O",                    # dicarboxylic
    ]
    ref_fps = []
    for smi in known_ndm1_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ref_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None or not ref_fps:
            scores.append(0.0)
            continue
        fp    = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        sims  = [DataStructs.TanimotoSimilarity(fp, r) for r in ref_fps]
        scores.append(round(float(np.mean(sims)) * 10, 4))  # scale 0-10
    return scores


# ── Combined Scoring ──────────────────────────────────────────────────────────
def combined_score(row) -> float:
    """
    Final score = 0.5 * DTI_score + 0.5 * similarity_score (both 0-10)
    Lipinski-failing drugs get a 50% penalty.
    """
    dti  = float(row.get('dti_score', 0) or 0)
    sim  = float(row.get('best_sim_score', 0) or 0) * 10  # scale to 0-10
    base = 0.5 * dti + 0.5 * sim

    if not row.get('pass_ro5', True):
        base *= 0.5   # penalise non-drug-like
    return round(base, 4)


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("=" * 55)
    print("  AGENT 3 — SCORING AGENT")
    print("=" * 55)

    # Load retrieval results
    if not os.path.exists(RETRIEVAL_RESULTS_CSV):
        raise FileNotFoundError(
            f"Not found: {RETRIEVAL_RESULTS_CSV}\n"
            "Run agent2_retrieval.py first."
        )
    df = pd.read_csv(RETRIEVAL_RESULTS_CSV)
    print(f"\n  Candidate drugs loaded: {len(df)}")

    # ── Step 1: Lipinski filter ───────────────────────────────────────────────
    print("\n[1] Applying Lipinski Rule-of-5...")
    lipinski_results = df['parent_smiles'].apply(lipinski_filter)
    df = pd.concat([df, pd.DataFrame(list(lipinski_results))], axis=1)
    pass_ro5 = df['pass_ro5'].sum()
    print(f"  Pass Ro5 : {pass_ro5} / {len(df)}")
    print(f"  Fail Ro5 : {len(df) - pass_ro5}")

    # ── Step 2: DTI scoring ───────────────────────────────────────────────────
    print("\n[2] DTI binding score (DeepPurpose vs NDM-1)...")
    scores = dti_score_deepurpose(df['parent_smiles'].tolist())
    df['dti_score'] = scores
    print(f"  DTI scoring done. Score range: "
          f"{min(scores):.3f} – {max(scores):.3f}")

    # ── Step 3: Combined score + rank ─────────────────────────────────────────
    print("\n[3] Computing combined score and ranking...")
    df['final_score'] = df.apply(combined_score, axis=1)
    df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    # ── Save scored ───────────────────────────────────────────────────────────
    df.to_csv(SCORED_CANDIDATES_CSV, index=False)

    # ── Final ranked output (clean columns) ───────────────────────────────────
    final_cols = [
        'rank', 'parent_chembl_id', 'parent_name',
        'parent_smiles', 'final_score', 'dti_score',
        'best_sim_score', 'n_hits', 'pass_ro5',
        'mw', 'logp', 'hbd', 'hba',
        'matched_frags'
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    final_df   = df[final_cols].copy()
    final_df.to_csv(FINAL_RANKED_CSV, index=False)

    print(f"\n  Scored saved        : {SCORED_CANDIDATES_CSV}")
    print(f"  Final ranked saved  : {FINAL_RANKED_CSV}")
    print(f"\n  Top 10 Repurposing Candidates:")
    print(final_df[['rank', 'parent_chembl_id', 'parent_name',
                    'final_score', 'pass_ro5']].head(10).to_string(index=False))

    print("\n[✓] Agent 3 complete.\n")
    return final_df


if __name__ == "__main__":
    run()
