
from pathlib import Path
import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
RAG_DIR = PROJECT_DIR / "medbot_phase3_rag"
DOCUMENTS_DIR = RAG_DIR / "documents"
PROCESSED_DIR = RAG_DIR / "processed"
INDEX_DIR = RAG_DIR / "vector_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Assignment target: approximately 600–1000 tokens per document chunk.
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100

# Keep embedding inputs below the model's maximum sequence length.
EMBEDDING_WINDOW_TOKENS = 180
EMBEDDING_WINDOW_OVERLAP = 30


def split_token_windows(token_ids, size, overlap):
    windows = []
    start = 0

    while start < len(token_ids):
        end = min(start + size, len(token_ids))
        windows.append(token_ids[start:end])

        if end == len(token_ids):
            break

        start = end - overlap

    return windows



def split_into_sections(text):
    """Preserve headings, paragraphs, and symptom lists."""
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
    ]

    # Recognize genuine article headings rather than every short line.
    known_headings = {
        "key facts",
        "overview",
        "symptoms",
        "type 1 diabetes",
        "type 2 diabetes",
        "gestational diabetes",
        "impaired glucose tolerance and impaired fasting glycaemia",
        "prevention",
        "diagnosis and treatment",
        "who response",
        "references",
    }

    sections = []
    current_heading = ""
    current_lines = []

    def save_section():
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({
                    "heading": current_heading,
                    "text": content,
                })

    for line in lines:
        if not line:
            save_section()
            current_lines = []
            continue

        normalized = line.lower().strip(" :")

        # Only treat recognized headings or clear heading-style questions
        # as headings. Preserve short symptom items as content.
        is_heading = (
            normalized in known_headings
            or (
                len(line) <= 100
                and line.endswith("?")
                and normalized.startswith(("what ", "how ", "why "))
            )
        )

        if is_heading:
            save_section()
            current_lines = []
            current_heading = line
        else:
            current_lines.append(line)

    save_section()

    if not sections and text.strip():
        sections.append({
            "heading": "",
            "text": text.strip(),
        })

    return sections



def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embedding model:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = model.tokenizer

    documents = []
    embedding_texts = []
    embedding_metadata = []

    source_files = sorted(DOCUMENTS_DIR.glob("document_*.json"))

    if not source_files:
        raise FileNotFoundError(
            f"No source documents found in {DOCUMENTS_DIR}"
        )

    for path in source_files:
        with path.open("r", encoding="utf-8") as file:
            source_document = json.load(file)

        text = source_document.get("text", "").strip()
        if not text:
            print("Skipping empty document:", path.name)
            continue

        sections = split_into_sections(text)
        chunk_number = 0

        for section in sections:
            heading = section["heading"]
            section_text = section["text"]

            # Include the heading in the searchable text.
            full_section_text = (
                f"{heading}\n{section_text}".strip()
                if heading else section_text
            )

            token_ids = tokenizer.encode(
                full_section_text,
                add_special_tokens=False,
                truncation=False,
            )

            # Split long sections into smaller overlapping chunks.
            token_windows = split_token_windows(
                token_ids,
                CHUNK_SIZE_TOKENS,
                CHUNK_OVERLAP_TOKENS,
            )

            for token_window in token_windows:
                chunk_text = tokenizer.decode(
                    token_window,
                    skip_special_tokens=True,
                ).strip()

                if not chunk_text:
                    continue

                chunk_number += 1
                chunk_id = f"{path.stem}_chunk_{chunk_number:03d}"

                record = {
                    "chunk_id": chunk_id,
                    "title": source_document["title"],
                    "source": source_document["source"],
                    "url": source_document["url"],
                    "heading": heading,
                    "chunk_number": chunk_number,
                    "token_count": len(token_window),
                    "text": chunk_text,
                }
                documents.append(record)

                subwindows = split_token_windows(
                    token_window,
                    EMBEDDING_WINDOW_TOKENS,
                    EMBEDDING_WINDOW_OVERLAP,
                )

                for subwindow_number, subwindow_ids in enumerate(
                    subwindows, start=1
                ):
                    embedding_text = tokenizer.decode(
                        subwindow_ids,
                        skip_special_tokens=True,
                    ).strip()

                    if not embedding_text:
                        continue

                    embedding_texts.append(embedding_text)
                    embedding_metadata.append({
                        **record,
                        "embedding_window": subwindow_number,
                        "text": embedding_text,
                        "embedding_token_count": len(subwindow_ids),
                    })

        print(
            f"{path.name}: {len(sections)} sections processed"
        )

    if not documents or not embedding_texts:
        raise RuntimeError("No document chunks or embedding windows created.")

    chunks_path = PROCESSED_DIR / "medical_chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as file:
        for record in documents:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nDocument chunks:", len(documents))
    print("Embedding windows:", len(embedding_texts))
    print("Generating embeddings...")

    embeddings = model.encode(
        embedding_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "medical.index"))

    with (INDEX_DIR / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            embedding_metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        "source_documents": len(source_files),
        "document_chunks": len(documents),
        "embedding_windows_indexed": len(embedding_metadata),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": int(embeddings.shape[1]),
        "document_chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "document_chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "embedding_window_tokens": EMBEDDING_WINDOW_TOKENS,
        "embedding_window_overlap_tokens": EMBEDDING_WINDOW_OVERLAP,
        "chunking_strategy": "heading and paragraph aware",
        "index_type": "FAISS IndexFlatIP with normalized embeddings",
    }

    with (INDEX_DIR / "index_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nVector index rebuilt successfully.")
    print("Document chunks:", chunks_path)
    print("FAISS index:", INDEX_DIR / "medical.index")
    print("Metadata:", INDEX_DIR / "metadata.json")
    print("Summary:", INDEX_DIR / "index_summary.json")


if __name__ == "__main__":
    main()
