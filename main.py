from app.rag_chain import create_rag_chain


def main():

    print("Starting RAG chatbot...")

    rag_chain = create_rag_chain()

    print("RAG chatbot is ready!")
    print("Type 'exit' to quit.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        if not question:
            continue

        response = rag_chain.invoke({
            "input": question
        })

        answer = response["answer"]

        print(f"\nAI: {answer}")


if __name__ == "__main__":
    main()