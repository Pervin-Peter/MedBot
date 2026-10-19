
"""
Automated answer tests for MedBot RAG.
Uses the existing retrieval and answer-generation functions.
"""

from medbot_rag_answer import (
    load_retrieval_components,
    load_medbot_model,
    retrieve_evidence,
    generate_answer,
)

TEST_QUESTIONS = [
    {
        "question": "What are the symptoms of diabetes?",
        "expected": ["thirsty", "urinating", "medlineplus.gov"],
    },
    {
        "question": "How can I prevent type 2 diabetes?",
        "expected": ["prevent", "exercise", "healthy diet"],
    },
    {
        "question": "What are the symptoms of asthma?",
        "expected": ["chest tightness", "wheezing", "medlineplus.gov"],
    },
    {
        "question": "What can trigger asthma?",
        "expected": ["allergens", "cold air", "medlineplus.gov"],
    },
    {
        "question": "What are the symptoms of high blood pressure?",
        "expected": ["no symptoms", "blood pressure", "medlineplus.gov"],
    },
    {
        "question": "What are the symptoms of migraine?",
        "expected": ["could not find medical evidence", "migraine"],
    },
]


def main():
    print("=" * 65)
    print("MEDBOT RAG AUTOMATED ANSWER TESTS")
    print("=" * 65)

    print("\nLoading retrieval components...")
    index, metadata, embedding_model = load_retrieval_components()

    print("Loading MedBot model...")
    tokenizer, model = load_medbot_model()

    passed = 0
    failed = 0

    for number, test in enumerate(TEST_QUESTIONS, start=1):
        question = test["question"]

        print("\n" + "=" * 65)
        print(f"TEST {number}/{len(TEST_QUESTIONS)}: {question}")
        print("=" * 65)

        try:
            evidence = retrieve_evidence(
                question, index, metadata, embedding_model
            )

            answer = generate_answer(
                question, evidence, tokenizer, model
            )

            print("\nANSWER:")
            print(answer)

            answer_lower = answer.lower()
            missing = [
                term
                for term in test["expected"]
                if term.lower() not in answer_lower
            ]

            if not missing:
                print("\nRESULT: PASS")
                passed += 1
            else:
                print(f"\nRESULT: CHECK - expected terms not found: {missing}")
                failed += 1

        except Exception as error:
            print(f"\nRESULT: FAIL - {type(error).__name__}: {error}")
            failed += 1

    print("\n" + "=" * 65)
    print("AUTOMATED TEST SUMMARY")
    print("=" * 65)
    print(f"Passed: {passed}")
    print(f"Check/Failed: {failed}")
    print(f"Total: {len(TEST_QUESTIONS)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
