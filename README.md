# Pokémon RAG Assistant

A retrieval-augmented generation (RAG) system that answers natural-language
questions about Sinnoh-region Pokémon. Built as a hands-on project to learn
how RAG works end to end: ingestion, chunking, embedding, semantic retrieval,
and grounded generation with citations.

The more important outcome than the tool itself was learning **when RAG is and
isn't the right approach** (see Reflections below).

## What it does

Ask questions in a terminal and get answers grounded in a local vector
database of Pokémon data:

Example question: What Pokémon most resembles a tree?
Example output: Sudowoodo most resembles a tree. According to its Pokédex description,
it pretends to be a tree despite being closer to rock than plant.
(Source: sudowoodo)


## Architecture

The system has two phases, a standard RAG split:

**Ingestion (`build_index.py`, run once):**
- Pulls the Sinnoh Pokédex list from the public PokéAPI
- For each Pokémon, fetches game data (types, stats, abilities) and species
  data (flavor text) from two endpoints and combines them
- Converts the structured data into a natural-language text chunk, because
  embedding models reason about prose far better than raw JSON
- One Pokémon per chunk (the data's natural semantic unit), stored in a
  persistent ChromaDB collection

**Query (`ask.py`, the CLI):**
- Embeds the user's question and retrieves the most similar chunks (semantic
  search)
- Injects the retrieved chunks into the system prompt as reference material
- Generates a grounded answer with citations back to the source Pokémon
- Instructed to answer only from retrieved material and to admit when the
  material is insufficient rather than guessing

## Tech

- **PokéAPI** — public data source (no scraping)
- **ChromaDB** — vector store with default `all-MiniLM-L6-v2` embeddings
- **Anthropic Claude (Haiku 4.5)** — generation; chosen deliberately as a
  small, cheap model since the task is simple
- **Python**

## What I tuned: the top-k experiment

Top-k (how many chunks retrieval returns) turned out to be the one knob that
genuinely mattered for this corpus. Chunk size and overlap were fixed by the
one-Pokémon-per-chunk structure, so there was nothing to tune there. Testing
the same questions across k = 1, 3, and 8 produced a clear, question-dependent
finding:

- **Single-answer semantic questions** ("most like a tree") are satisfied at
  k=1; higher k is wasted cost with no quality gain.
- **Survey / superlative questions** ("fastest," "best") need higher k,
  because the answer is a comparison across many candidates and low k starves
  it.
- **The sharpest finding:** for "best of" questions, insufficient k didn't just
  weaken the answer, it *changed* it. The same "single best Pokémon" question
  returned Lucario at k=3 and Gyarados at k=8, both well-reasoned given their
  context, because the true best candidate simply wasn't retrieved at low k.
  Retrieval silently bounds the correctness ceiling of everything downstream.

Based on this, the CLI defaults to k=5: missing the right candidate (a wrong
answer) is worse than mild dilution from a few extra chunks.

## Honest limitations

Testing surfaced real boundaries, which were the most instructive part:

- **No reasoning over missing data.** Questions like "which Pokémon has the
  largest stat change across its evolution" can't be answered, because
  evolution-chain relationships were never ingested. No top-k setting fixes a
  missing-data problem, though the system fails honestly rather than inventing.
- **Retrieval finds topical similarity, not logical answers.** Asked "best
  Pokémon for a grass gym," it retrieved Pokémon *about* grass (including Grass
  types that would lose) rather than Pokémon that *beat* grass. Embeddings match
  meaning, they don't reason about type effectiveness.
- **No type-effectiveness or legendary data.** These aren't per-Pokémon prose;
  type effectiveness is a global lookup table and legendary status is a
  structured field. Forcing them into RAG would be the wrong tool. The right
  design is **hybrid**: semantic retrieval for prose, plus structured lookups
  for relational/tabular data.

## Reflections: when RAG is the wrong tool

The most valuable lesson was recognizing that **Pokémon is a poor real-world
RAG candidate.** The data is public, static, and thoroughly represented in any
model's training knowledge, so a production tool would be better served by the
model's own knowledge plus a web-search tool, with far less engineering.

RAG earns its place when data is **private, rapidly changing, or requires
auditable citations**, which is something that Pokemon doesn't need. Building this made the
distinction concrete: the reflexive reach for RAG is a common and expensive
over-engineering mistake, and the real skill is knowing when a simpler approach
wins.

Pokémon was nonetheless the right choice *for learning*, precisely because I
know the domain well enough to immediately judge when retrieval returned
something wrong, which is exactly what made the failures above visible.

## Running it
pip install anthropic chromadb requests python-dotenv
python build_index.py # one-time: builds the vector database
python ask.py # interactive Q&A
Requires an Anthropic API key in a `.env` file (not included).