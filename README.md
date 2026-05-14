# Agentic-AI-for-Drug-Repurposing
An Agentic AI framework for identifying potential drug candidates against antimicrobial resistance (AMR) targets such as NDM-1 using fragment-based retrieval, molecular embeddings, and drug–target interaction prediction.
This project combines:

🧩 BRICS Fragmentation for substructure extraction
🧠 ChemBERTa embeddings for molecular representation
🔍 FAISS + KNN retrieval for fragment similarity search
⚗️ DeepPurpose DTI models for binding affinity prediction
🛡️ ADMET filtering for drug-likeness and safety analysis
🤖 Agentic AI orchestration for autonomous retrieval → scoring → explanation workflows
🚀 Objective

Instead of comparing whole molecules directly, the system uses biologically meaningful fragments from known inhibitors to retrieve structurally relevant approved drugs and evaluate them as repurposing candidates for AMR-related targets.

🧠 Key Features
Fragment-guided drug retrieval
Embedding-based molecular similarity search
Target-driven autonomous repurposing pipeline
Drug–Target Interaction (DTI) scoring
Explainable AI workflow for candidate reasoning
Modular multi-agent architecture
⚙️ Tech Stack
Component	Tools / Models
Fragmentation	RDKit (BRICS)
Embeddings	ChemBERTa
Retrieval	FAISS + KNN
DTI Prediction	DeepPurpose
Filtering	ADMET-AI
Agent Framework	CrewAI / LangGraph
UI	Streamlit
🧪 Example Workflow
Target (NDM-1)
    ↓
Known inhibitors from ChEMBL
    ↓
Fragment extraction
    ↓
Embedding generation
    ↓
Similarity retrieval over approved drugs
    ↓
DTI scoring + ADMET filtering
    ↓
Ranked repurposed drug candidates
📚 Datasets Used
ChEMBL
DrugBank
BindingDB
PDB (3S0Z – NDM-1 structure)
ADMET benchmarks
ChemBERTa pretrained embeddings
🔬 Research Focus

This project explores:

AI-driven antimicrobial drug discovery
Fragment-based drug repurposing
Molecular foundation models
Agentic AI for biomedical research workflows
Explainable computational pharmacology
📌 Status

🚧 Research & Development Phase
Planned extensions include:

Multi-target AMR support
PubMed RAG integration
Iterative autonomous agent loops
Docking-based validation
Explainability dashboards
