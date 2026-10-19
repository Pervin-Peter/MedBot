
import argparse
import html
import json
import re
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split


DATASET_NAME = "AnonymousSub/MedQuAD_47441_Question_Answer_Pairs"
OUTPUT_DIR = Path("medbot_phase1_output")


def clean_text(value):
    """Remove HTML, decode entities, and normalize whitespace."""
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_column(columns, candidates):
    """Find a matching dataset column name."""
    lookup = {column.lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    for column in columns:
        if any(candidate.lower() in column.lower()
               for candidate in candidates):
            return column

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional small sample for testing; omit for full dataset."
    )
    args = parser.parse_args()

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    # Use the first available split for initial dataset inspection.
    split_name = next(iter(dataset.keys()))
    data = dataset[split_name]

    if args.max_rows:
        data = data.select(range(min(args.max_rows, len(data))))

    columns = data.column_names
    print(f"Available columns: {columns}")
    print(f"Rows loaded: {len(data)}")

    question_col = find_column(
        columns, ["question", "query", "instruction"]
    )
    answer_col = find_column(
        columns, ["answer", "response", "output"]
    )

    if not question_col or not answer_col:
        raise ValueError(
            "Could not identify question/answer columns. "
            f"Available columns: {columns}. "
            "Update the column candidates in the script."
        )

    records = []
    seen = set()
    empty_rows = 0
    duplicate_rows = 0

    for row in data:
        question = clean_text(row.get(question_col))
        answer = clean_text(row.get(answer_col))

        if not question or not answer:
            empty_rows += 1
            continue

        pair = (question.casefold(), answer.casefold())
        if pair in seen:
            duplicate_rows += 1
            continue

        seen.add(pair)
        records.append({
            "instruction": question,
            "input": "",
            "output": answer
        })

    if len(records) < 2:
        raise ValueError(
            "Not enough valid records to create a train/validation split."
        )

    train_records, validation_records = train_test_split(
        records,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(OUTPUT_DIR / "medquad_cleaned.jsonl", records)
    write_jsonl(OUTPUT_DIR / "medquad_train.jsonl", train_records)
    write_jsonl(
        OUTPUT_DIR / "medquad_validation.jsonl", validation_records
    )

    summary = {
        "dataset": DATASET_NAME,
        "source_split": split_name,
        "source_rows_processed": len(data),
        "question_column": question_col,
        "answer_column": answer_col,
        "valid_unique_records": len(records),
        "empty_rows_removed": empty_rows,
        "duplicate_rows_removed": duplicate_rows,
        "training_records": len(train_records),
        "validation_records": len(validation_records),
        "validation_ratio": 0.20,
        "random_seed": 42
    }

    with (OUTPUT_DIR / "preprocessing_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nPreprocessing completed.")
    print(json.dumps(summary, indent=2))
    print(f"\nOutput folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
