"""CLI entry point for BM Policy Analyzer."""
from src.query import query_documents


def main():
    print("BM Technology Handbook Q&A")
    print("=" * 40)
    print("Ask questions about company policies.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("Question: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        answer, docs, distances = query_documents(
            user_input,
            top_k=5,
            use_transform=True,
            similarity_threshold=0.3
        )

        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()