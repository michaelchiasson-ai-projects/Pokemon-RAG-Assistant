import os
import chromadb
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()
model = "claude-haiku-4-5-20251001"

chroma_client = chromadb.PersistentClient(path="./pokemon_db")
collection = chroma_client.get_collection(name="pokemon")

def retrieve(query, n_results=5):
    results = collection.query(query_texts=[query], n_results=n_results)
    return list(zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0],
    ))

def answer_question(query, n_results=5):
    matches = retrieve(query, n_results)
    context = "\n\n".join(f"[Source: {pid}]\n{doc}" for pid, doc, _ in matches)
    system_prompt = f"""You are a knowledgeable Pokemon assistant. Answer the user's question using ONLY the reference material below.

Important rules:
- Base your answer only on the reference material. Do not use outside Pokemon knowledge.
- If the reference material does not contain enough information, say so plainly rather than guessing.
- When you state a fact about a specific Pokemon, cite it like (Source: garchomp).
- If the retrieved Pokemon are a poor fit for the question, say that honestly.

<reference_material>
{context}
</reference_material>"""
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )
    return resp.content[0].text, matches

def main():
    print("=" * 55)
    print("  Pokemon RAG Assistant")
    print("  Ask questions about Sinnoh Pokemon.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 55)

    while True:
        query = input("\nYour question: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not query:
            print("Please type a question, or 'quit' to exit.")
            continue

        print("\nThinking...\n")
        answer, matches = answer_question(query)

        print("-" * 55)
        print(answer)
        print("-" * 55)
        print("Sources retrieved:", ", ".join(pid for pid, _, _ in matches))

if __name__ == "__main__":
    main()