MEDBOT — COMPLETE SUBMISSION README
SRM Institute of Science and Technology
Course: 21CSE658T Large Language Models | LLT Assessment — Set B
Project: MedBot: Building an Intelligent Medical Q&A Assistant using Fine-Tuning, RAG, and Tool Integration
Student: Pervin Peter P. | Register No.: RA2512049015027

PROJECT ROOT
D:\SRM\Sem-3\Large Language Models\MedBot

RUN THE STREAMLIT APP
1. Open PowerShell in the project root.
2. Activate the virtual environment if it is not already active:
   .venv\Scripts\Activate.ps1
3. Start the app:
   python -m streamlit run app.py
4. Open the localhost URL printed by Streamlit (normally http://localhost:8501).
5. Stop with Ctrl+C in the terminal.

PROJECT SUMMARY
Phase 1 — Dataset preparation
- Source rows: 47,441
- Clean valid unique records: 16,359
- Train: 13,087; validation: 3,272 (80:20)
- Outputs: medquad_cleaned.jsonl, medquad_train.jsonl, medquad_validation.jsonl, preprocessing_summary.json, preprocessing report.

Phase 2 — Fine-tuning
- Base model: Qwen/Qwen2.5-1.5B-Instruct
- 4-bit NF4 QLoRA using PEFT
- LoRA targets: q_proj, k_proj, v_proj, o_proj
- Training loss: 1.1681; evaluation loss: 1.1514
- Adapter path: medbot_phase2_full/lora_adapter
- Note: losses are not generation-quality or clinical-safety metrics.

Phase 3 — RAG
- Five public medical documents from MedlinePlus/NIH, WHO, and CDC covering diabetes, high blood pressure, and asthma.
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
- FAISS IndexFlatIP with normalized embeddings.
- Index summary: 41 document chunks and 81 embedding windows.
- Configured document chunks: 800 tokens with 100-token overlap.
- Embedding windows: 180 tokens with 30-token overlap.
- History-aware follow-up rewriting is implemented in make_history_aware_query() and used before evidence retrieval.

Phase 5 — Tools
- Local BMI calculator; tested example: 70 kg, 170 cm -> BMI 24.2.
- FDA DrugInfo lookup: https://api.fda.gov/drug/label.json; ibuprofen label lookup displayed information and a source URL.
- FDA label content is not individualized dosing advice.

Phase 6 — User interface
- Streamlit medical Q&A
- Retrieved evidence cards and source URLs
- Similarity indicators and medical disclaimer
- BMI and FDA lookup forms
- History stored in Streamlit session state for follow-up rewriting

KEY FILES
- app.py
- medbot_rag_answer.py
- preprocess_medquad.py
- train_medbot_full.py
- collect_medical_documents.py
- build_medbot_vector_index.py
- test_medbot_retrieval.py
- test_medbot_rag_answers.py
- medbot_phase1_output/
- medbot_phase2_full/lora_adapter/
- medbot_phase3_rag/documents/
- medbot_phase3_rag/processed/
- medbot_phase3_rag/vector_index/

TESTS REPORTED
- Defined automated RAG answer tests: 6 passed, 0 failed/check.
- Diabetes symptoms: source evidence displayed.
- Asthma symptoms: source evidence displayed.
- Asthma follow-up: “How can I prevent it?” resolves to “How can I prevent asthma?”
- Unsupported XYZ-123 condition: refusal verified in direct Python and Streamlit.
- BMI: 70 kg / 170 cm -> 24.2, Normal weight.
- FDA lookup: ibuprofen label information displayed.

REQUIREMENT COVERAGE / LIMITATIONS
- Training/evaluation loss is recorded, but ROUGE-L/BERTScore generation evaluation was not confirmed.
- Required finetune_medbot.ipynb, rag_medbot.ipynb, and medbot_agent.ipynb were not independently verified. The existing implementation uses scripts and app.py. If the exact notebooks are mandatory, ensure they are present before submission.
- The assignment specifies LangChain or LlamaIndex; the current RAG implementation uses Sentence Transformers and FAISS directly.
- The app rewrites follow-up questions using chat history, but a unified tool-call trace and model-directed tool invocation were not confirmed.
- Asthma prevention follow-up can receive a treatment-focused answer. Evidence/intent alignment remains a limitation.
- Low-similarity retrieved passages may appear alongside an unsupported-question refusal.
- RLHF/GRPO was optional and not implemented; Disease.sh was not implemented because the API example was optional.
- The current medical corpus contains five documents and has limited topic coverage.
- Do not claim clinical validation.

SAFETY
MedBot is an educational prototype, not a diagnostic system. It does not replace a qualified healthcare professional and should not be used for self-diagnosis or individualized treatment decisions.

ENVIRONMENT REPORTED DURING TESTING
Python 3.13.13; PyTorch 2.11.0+cu128; Transformers 5.20.1; PEFT 0.21.2; Accelerate 1.15.0; bitsandbytes 0.50.2; datasets 5.1.0; sentence-transformers 6.1.0; faiss-cpu 1.15.1.

REFERENCES
MedQuAD: https://huggingface.co/datasets/AnonymousSub/MedQuAD_47441_Question_Answer_Pairs
Qwen2.5-1.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
MedlinePlus: https://medlineplus.gov/
WHO: https://www.who.int/
CDC: https://www.cdc.gov/
openFDA: https://api.fda.gov/drug/label.json
FAISS: https://github.com/facebookresearch/faiss
PEFT: https://huggingface.co/docs/peft/
Streamlit: https://docs.streamlit.io/
