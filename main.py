"""CLI entry point for BM Policy Analyzer."""
from src.query import query_documents


def main():
    print("BM Technology Handbook Q&A")
    print("=" * 40)
    print("Ask questions about company policies.\n")

    test_question = "What is the vacation policy?"
    print(f"Testing with: {test_question}\n")

    answer, docs, distances = query_documents(
        test_question,
        top_k=5,
        use_transform=False,
        similarity_threshold=0.3
    )

    print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()