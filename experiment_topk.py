import os
import chromadb
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()
model = "claude-haiku-4-5-20251001"

chroma_client = chromadb.PersistentClient(path="./pokemon_db")
collection = chroma_client.get_collection(name="pokemon")

def retrieve(query, n_results):
    results = collection.query(query_texts=[query], n_results=n_results)
    return list(zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0],
    ))

def answer_with_k(query, k):
    matches = retrieve(query, k)
    context = "\n\n".join(f"[Source: {pid}]\n{doc}" for pid, doc, _ in matches)
    system_prompt = f"""You are a Pokemon assistant. Answer using ONLY the reference material below. Cite facts with the source name like (Source: garchomp). If the material is insufficient, say so.

<reference_material>
{context}
</reference_material>"""
    resp = client.messages.create(
        model=model,
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )
    return resp.content[0].text, matches

if __name__ == "__main__":
    query = input("Question to test across k values: ")
    for k in [1, 3, 8]:
        print(f"\n{'='*60}")
        print(f"  k = {k}")
        print(f"{'='*60}")
        answer, matches = answer_with_k(query, k)
        print("Retrieved:", ", ".join(pid for pid, _, _ in matches))
        print("\nAnswer:")
        print(answer)