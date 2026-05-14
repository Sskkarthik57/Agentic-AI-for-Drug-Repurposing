"""
config.py — Central configuration for all agents and scripts.
All paths, constants, and settings in one place.
"""

import os
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Folder paths ─────────────────────────────────────────────────────────────
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
AGENTS_DIR  = ROOT / "agents"
PHASE2_DIR  = ROOT / "phase2"
UTILS_DIR   = ROOT / "utils"

# ── Input files (place your CSVs in data/) ───────────────────────────────────
APPROVED_DRUGS_CSV     = DATA_DIR / "approved_drugs_chembl.csv"
NDM1_INHIBITORS_CSV    = DATA_DIR / "ndm1_inhibitors_all.csv"
NDM1_BENCHMARK_CSV     = DATA_DIR / "ndm1_benchmark_clean.csv"

# ── Output files (auto-generated) ────────────────────────────────────────────
FRAGMENTS_CSV          = OUTPUT_DIR / "drug_fragments.csv"
FAISS_INDEX_PATH       = OUTPUT_DIR / "fragment_index.faiss"
FRAGMENT_META_CSV      = OUTPUT_DIR / "fragment_metadata.csv"
RETRIEVAL_RESULTS_CSV  = OUTPUT_DIR / "retrieval_results.csv"
SCORED_CANDIDATES_CSV  = OUTPUT_DIR / "scored_candidates.csv"
FINAL_RANKED_CSV       = OUTPUT_DIR / "final_ranked_drugs.csv"
EVALUATION_REPORT      = OUTPUT_DIR / "evaluation_report.txt"
PHASE2_REPORT          = OUTPUT_DIR / "phase2_report.md"

# ── NDM-1 target configuration ───────────────────────────────────────────────
NDM1_TARGET_IDS = [
    "CHEMBL1667695",   # Klebsiella pneumoniae — primary
    "CHEMBL4295540",   # E. coli
    "CHEMBL2366517",   # variant
]

# NDM-1 protein sequence (UniProt C7C422) — used for DTI scoring
NDM1_SEQUENCE = (
    "MSLPKLSLFSLATAFASSSIAQAKELPQLGVSMEDLVARIRELRQANSDNPTVKLLYQDG"
    "QTFYELAKLAEQFGDQLVGLNILTEEGVHYSYDLHQQLAQRIGKQPDQSALFYQLAAQAG"
    "AQVEASASQRLKELARLHQAQFQELPELRGQISQLAQQLRTQVQDQPASEALAQFGDQSAA"
    "ELDNRLAAREARQAQIDMPSSIEAAVGLVAQMGAALAGHLKKLPGLREIESQRQEAEDQLA"
    "LARLLNQQSGQKASAEQARLSLQLAADAFADRFNQQLAQKIPGLSVNAQAVADYVAEYHALPAPAW"
)

# ── Fragmentation settings ────────────────────────────────────────────────────
MIN_FRAGMENT_ATOMS = 3       # discard trivially small fragments
MAX_FRAGMENT_ATOMS = 50      # discard unreasonably large fragments

# ── Embedding settings ────────────────────────────────────────────────────────
MORGAN_RADIUS      = 2       # Morgan fingerprint radius
MORGAN_NBITS       = 2048    # Morgan fingerprint bit length
CHEMBERTA_MODEL    = "seyonec/ChemBERTa-zinc-base-v1"
EMBEDDING_DIM      = 768     # ChemBERTa output dimension

# ── Retrieval settings ────────────────────────────────────────────────────────
TOP_K              = 20      # number of similar fragments to retrieve
EMBEDDING_TYPE     = "chemberta"   # "morgan" or "chemberta"

# ── Scoring / filtering settings ─────────────────────────────────────────────
ACTIVE_THRESHOLD_NM     = 10000    # IC50/Ki ≤ 10,000 nM = active
LIPINSKI_MW_MAX         = 500
LIPINSKI_LOGP_MAX       = 5
LIPINSKI_HBD_MAX        = 5
LIPINSKI_HBA_MAX        = 10

# ── Evaluation settings ───────────────────────────────────────────────────────
BENCHMARK_HOLDOUT_FRAC  = 0.2     # 20% holdout for evaluation
RECALL_K_VALUES         = [5, 10, 20]

# ── LLM settings (Phase 2) ───────────────────────────────────────────────────
PRIMARY_MODEL   = "llama-3.3-70b-versatile"
FALLBACK_MODEL  = "llama3-8b-8192"
MAX_REQUERY     = 3           # max re-query iterations in orchestrator
SCORE_THRESHOLD = 6.0         # min DTI score to accept without re-querying
