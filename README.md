# MedBot — Intelligent Medical Q&A Assistant

**SRM Institute of Science and Technology**  
**Course:** 21CSE658T — Large Language Models  
**Assessment:** LLT Assessment — Set B  
**Student:** Pervin Peter P.  
**Register Number:** RA2512049015027

MedBot is an academic project that combines a fine-tuned language model, Retrieval-Augmented Generation (RAG), medical source retrieval, and tool integration to answer medical questions with supporting evidence.

> **Medical disclaimer:** MedBot is an educational prototype, not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

## Project Objectives

- Prepare and clean medical question-and-answer data.
- Fine-tune an instruction-following language model using QLoRA.
- Retrieve relevant evidence from public medical information sources.
- Support follow-up questions using conversation history.
- Provide supporting source links and a medical disclaimer.
- Demonstrate tool integration through a BMI calculator and FDA drug-label lookup.

## System Architecture

```text
Medical Q&A Dataset
        |
        v
Data Cleaning and Preprocessing
        |
        v
Train / Validation Split
        |
        v
Qwen2.5-1.5B-Instruct + QLoRA Fine-Tuning
        |
        v
Fine-Tuned Adapter
        |
        +----------------------+
                               |
User Question --> Query Processing
                               |
                               v
                     Medical Evidence Retrieval
                               |
                               v
                     FAISS Vector Index
                               |
                               v
                     Evidence-Grounded Answer
                               |
                               v
                    Streamlit User Interface
                               |
                 Sources, Disclaimer and Tools
```

## Implementation

### Phase 1 — Dataset Preparation

- Source dataset rows: 47,441
- Clean, valid, unique records: 16,359
- Training records: 13,087
- Validation records: 3,272
- Split: 80:20

Generated datasets are excluded from this Git repository to avoid unnecessary data duplication. Run the preprocessing script to regenerate them from the required source dataset.

### Phase 2 — Fine-Tuning

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Method:** 4-bit NF4 QLoRA using PEFT
- **LoRA target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Training loss:** 1.1681
- **Evaluation loss:** 1.1514
- **Adapter output:** `medbot_phase2_full/lora_adapter`

Loss values measure training performance; they do not establish medical accuracy or clinical safety.

The trained adapter and training checkpoints are not included in this repository. The application requires the corresponding model artifacts to be available locally.

### Phase 3 — Retrieval-Augmented Generation

The RAG pipeline uses five public medical documents covering diabetes, high blood pressure, and asthma.

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding dimension:** 384
- **Vector index:** FAISS `IndexFlatIP` with normalized embeddings
- **Document chunks:** 41
- **Embedding windows:** 81
- **Document chunk configuration:** 800 tokens with 100-token overlap
- **Embedding window configuration:** 180 tokens with 30-token overlap

The implementation includes history-aware query rewriting to help interpret follow-up questions.

### Tools and User Interface

The Streamlit interface provides:

- Medical question-and-answer interaction
- Retrieved evidence and source URLs
- Similarity indicators and medical disclaimer
- BMI calculation
- FDA drug-label lookup using the openFDA API
- Session-based conversation history

The FDA lookup displays label information and is not an individualized medication recommendation.

## Test Results

The project testing record reports six automated RAG answer tests passed:

1. Diabetes symptoms
2. Type 2 diabetes prevention
3. Asthma symptoms
4. Asthma triggers
5. High blood pressure symptoms
6. Unsupported-question refusal

A BMI test using 70 kg and 170 cm returned a BMI of 24.2.

These are project-level functional checks, not a clinical validation study. Results should not be interpreted as proof of medical reliability.

## Project Structure

```text
MedBot/
├── app.py
├── medbot_rag_answer.py
├── preprocess_medquad.py
├── finetune_medbot.py
├── train_medbot_full.py
├── build_medbot_vector_index.py
├── collect_medical_documents.py
├── check_medbot_dataset.py
├── preflight_medbot.py
├── test_full_training.py
├── test_medbot_adapter.py
├── test_medbot_retrieval.py
├── test_medbot_rag_answers.py
├── medbot_phase1_output/
├── medbot_phase3_rag/
├── MedBot_Final_Submission_Report.docx
└── MedBot_COMPLETE_SUBMISSION_README.txt
```

The Phase 2 training output and large Phase 1 datasets are intentionally excluded from Git.

## Setup

### Prerequisites

- Python version compatible with the project's installed ML libraries
- Git
- GPU with sufficient memory for the configured quantized model, or an appropriately adapted inference setup
- Access to the required base model and its dependencies

### Install dependencies

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies after creating the project's `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

### Run the application

From the project root:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

**Note:** A fresh clone does not include the trained adapter or Phase 1 dataset files. Some application features may require these artifacts to be generated or downloaded before they can run. The exact setup should be verified on a fresh environment.

## Limitations and Future Work

- Medical coverage is limited by the available dataset and the five-document RAG collection.
- Training and evaluation loss do not establish clinical accuracy.
- The system requires broader evaluation for hallucination, retrieval quality, bias, and medical safety.
- The trained adapter is not included in this repository.
- Reinforcement learning / RLHF has not been implemented in the current version.
- Tool integration is demonstrated through the BMI and FDA lookup features; complete autonomous tool orchestration has not been established.
- Further work should include broader source coverage, stronger evaluation, improved refusal behavior, and clinical expert review.

## References and Data Sources

- [MedlinePlus](https://medlineplus.gov/)
- [World Health Organization](https://www.who.int/)
- [Centers for Disease Control and Prevention](https://www.cdc.gov/)
- [openFDA Drug Label API](https://open.fda.gov/apis/drug/label/)

Refer to the source documents and manifest included in `medbot_phase3_rag/` for the collection used by this project.

## Academic Use

This project was developed as part of the SRM Institute of Science and Technology Large Language Models course assessment. Training metrics and functional tests are reported for academic transparency; independent clinical validation has not been performed.