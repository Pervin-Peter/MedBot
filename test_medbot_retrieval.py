
from pathlib import Path
import json
import re

import faiss
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
RAG_DIR = PROJECT_DIR / "medbot_phase3_rag"
INDEX_DIR = RAG_DIR / "vector_index"

INDEX_PATH = INDEX_DIR / "medical.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 50
MAX_RESULTS = 10
MAX_PASSAGE_CHARS = 1000


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def question_type(question):
    q = question.lower()

    if any(word in q for word in ["symptom", "signs", "feel"]):
        return "symptoms"

    if any(word in q for word in ["treat", "treatment", "cure", "manage"]):
        return "treatment"

    if any(word in q for word in ["cause", "causes", "caused by"]):
        return "causes"

    if any(word in q for word in ["prevent", "prevention", "avoid"]):
        return "prevention"

    if any(word in q for word in ["diagnos", "test", "measure", "reading"]):
        return "diagnosis"

    return "general"



def evidence_bonus(text, kind, question):
    """Reward direct evidence only when it matches the requested condition."""
    text = clean_text(text).lower()
    question_lower = question.lower()
    bonus = 0.0
    reasons = []

    # Identify the condition being asked about.
    condition_terms = []

    if "high blood pressure" in question_lower or "hypertension" in question_lower:
        condition_terms = [
            "high blood pressure",
            "hypertension",
            "blood pressure",
        ]
    elif "diabetes" in question_lower:
        condition_terms = ["diabetes", "blood sugar"]
    elif "asthma" in question_lower:
        condition_terms = ["asthma"]
    elif "heart disease" in question_lower:
        condition_terms = ["heart disease"]
    elif "kidney disease" in question_lower:
        condition_terms = ["kidney disease"]

    # Penalize passages that clearly focus on a different condition.
    title = ""
    for term in condition_terms:
        if term in text:
            bonus += 0.08
            reasons.append(f"condition match: {term}")
            break

    # This question specifically concerns hypertension symptoms.
    if kind == "symptoms" and condition_terms and any(
        phrase in text
        for phrase in [
            "usually has no symptoms",
            "often has no symptoms",
            "high blood pressure has no symptoms",
        ]
    ):
        if any(term in text for term in condition_terms):
            bonus += 0.30
            reasons.append("direct evidence: symptoms may be absent")

    # Give a smaller reward to passages directly discussing symptoms.
    if kind == "symptoms" and any(
        phrase in text
        for phrase in [
            "symptoms of high blood pressure",
            "symptoms of hypertension",
            "signs of high blood pressure",
            "signs of hypertension",
        ]
    ):
        bonus += 0.15
        reasons.append("direct symptom discussion")

    # Match topic terms only after checking the condition.
    topic_terms = {
        "symptoms": ["symptom", "signs and symptoms"],
        "treatment": ["treatment", "treated with", "medication"],
        "causes": ["caused by", "causes of", "risk factors"],
        "prevention": ["prevent", "prevention", "reduce your risk"],
        "diagnosis": ["diagnosed", "diagnosis", "blood pressure checks"],
        "general": [],
    }

    if condition_terms and any(term in text for term in condition_terms):
        if any(term in text for term in topic_terms[kind]):
            bonus += 0.08
            reasons.append("condition-specific topic match")

    # Strongly penalize unrelated disease passages for condition-specific questions.
    if condition_terms and not any(term in text for term in condition_terms):
        if any(term in text for term in ["diabetes", "asthma", "migraine"]):
            bonus -= 0.25
            reasons.append("unrelated-condition penalty")

    return bonus, reasons



def main():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "FAISS index or metadata file is missing."
        )

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index has {index.ntotal} vectors, "
            f"but metadata has {len(metadata)} records."
        )

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    question = input("\nEnter a medical question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    kind = question_type(question)

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    search_count = min(TOP_K, index.ntotal)
    scores, indices = index.search(query_embedding, search_count)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = metadata[int(idx)]
        passage = clean_text(item.get("text", ""))

        bonus, reasons = evidence_bonus(passage, kind, question)
        final_score = float(score) + bonus

        candidates.append({
            "similarity": float(score),
            "final_score": final_score,
            "bonus": bonus,
            "reasons": reasons,
            "item": item,
            "passage": passage,
        })

    # Rank using semantic similarity plus topic-specific evidence.
    candidates.sort(
        key=lambda result: result["final_score"],
        reverse=True,
    )

    # Avoid returning multiple overlapping embedding windows
    # from the same parent document chunk.
    results = []
    seen_chunks = set()

    for candidate in candidates:
        item = candidate["item"]
        chunk_id = item.get(
            "chunk_id",
            item.get("url", item.get("title", "")),
        )

        if chunk_id in seen_chunks:
            continue

        seen_chunks.add(chunk_id)
        results.append(candidate)

        if len(results) >= MAX_RESULTS:
            break

    print("\nQuestion:", question)
    print("Question type:", kind)
    print("\nRetrieved and reranked evidence")
    print("=" * 75)

    if not results:
        print("No relevant passages were retrieved.")
        return

    for rank, result in enumerate(results, start=1):
        item = result["item"]
        passage = result["passage"]

        if len(passage) > MAX_PASSAGE_CHARS:
            passage = passage[:MAX_PASSAGE_CHARS].rsplit(" ", 1)[0] + "..."

        print(f"\nResult {rank}")
        print(f"Similarity score: {result['similarity']:.4f}")
        print(f"Reranked score: {result['final_score']:.4f}")
        print(f"Evidence priority: {', '.join(result['reasons']) or 'Semantic similarity'}")
        print(f"Title: {item.get('title', 'Unknown')}")
        print(f"Section: {item.get('heading', 'Not recorded')}")
        print(f"Source: {item.get('source', 'Unknown')}")
        print(f"URL: {item.get('url', 'Not available')}")
        print("\nPassage:")
        print(passage)
        print("-" * 75)


if __name__ == "__main__":
    main()
