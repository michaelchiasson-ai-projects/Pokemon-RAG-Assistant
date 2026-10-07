import time
import requests
import chromadb

# ---------------------------------------------------------------------------
# The starter batch: Garchomp plus 19 other Sinnoh Pokemon.
# PokeAPI names are lowercase and hyphenated.
# ---------------------------------------------------------------------------
def get_sinnoh_pokemon_names():
    """
    Fetch the full Sinnoh Pokedex species list from PokeAPI.
    The 'original-sinnoh' pokedex endpoint returns an index of entries,
    each giving a species name. This is the 'index endpoint' pattern:
    one call gives the list, then we fetch each entry's details separately.
    """
    resp = requests.get("https://pokeapi.co/api/v2/pokedex/original-sinnoh")
    resp.raise_for_status()  # fail loudly if this core call doesn't work
    data = resp.json()
    # Each entry has pokemon_species.name; pull them in dex order.
    names = [entry["pokemon_species"]["name"] for entry in data["pokemon_entries"]]
    return names

POKEMON_NAMES = get_sinnoh_pokemon_names()

# ---------------------------------------------------------------------------
# Fetch one Pokemon from both endpoints and format it into one text chunk.
# ---------------------------------------------------------------------------
def build_pokemon_chunk(name):
    """
    Fetch a Pokemon's game data and species data, then combine them into a
    single readable text blob. Returns (chunk_text, metadata) or None on failure.
    """
    # Game-mechanics data: types, stats, abilities.
    pokemon_resp = requests.get(f"https://pokeapi.co/api/v2/pokemon/{name}")
    if pokemon_resp.status_code != 200:
        print(f"  ! Could not fetch pokemon data for '{name}' (status {pokemon_resp.status_code})")
        return None
    pokemon_data = pokemon_resp.json()

    # Species data: the prose flavor text.
    species_resp = requests.get(f"https://pokeapi.co/api/v2/pokemon-species/{name}")
    if species_resp.status_code != 200:
        print(f"  ! Could not fetch species data for '{name}' (status {species_resp.status_code})")
        return None
    species_data = species_resp.json()

    # --- Pull the fields we care about out of the nested JSON ---
    display_name = pokemon_data["name"].capitalize()
    types = [t["type"]["name"] for t in pokemon_data["types"]]
    abilities = [a["ability"]["name"] for a in pokemon_data["abilities"]]
    stats = {s["stat"]["name"]: s["base_stat"] for s in pokemon_data["stats"]}

    # First English flavor text (there are many; we take one and tidy it).
    flavor = ""
    for entry in species_data["flavor_text_entries"]:
        if entry["language"]["name"] == "en":
            # Flavor text has stray newlines and form-feed characters; clean them.
            flavor = entry["flavor_text"].replace("\n", " ").replace("\f", " ").strip()
            break

    # --- Turn the structured data into natural-language text ---
    # We write it as prose because the embedding model reasons about text,
    # not raw JSON. This is the structured-to-natural-language translation step.
    type_str = " and ".join(t.capitalize() for t in types)
    ability_str = ", ".join(a.replace("-", " ") for a in abilities)

    chunk_text = (
        f"{display_name} is a {type_str} type Pokemon. "
        f"Its base stats are: HP {stats['hp']}, Attack {stats['attack']}, "
        f"Defense {stats['defense']}, Special Attack {stats['special-attack']}, "
        f"Special Defense {stats['special-defense']}, Speed {stats['speed']}. "
        f"Its abilities are {ability_str}. "
        f"Pokedex description: {flavor}"
    )

    # Metadata travels alongside the chunk; useful for filtering later.
    metadata = {
        "name": display_name,
        "types": ",".join(types),  # Chroma metadata values must be simple types
    }

    return chunk_text, metadata

# ---------------------------------------------------------------------------
# Build the whole index.
# ---------------------------------------------------------------------------
def main():
    # Persistent Chroma store, saved to ./pokemon_db on disk.
    chroma_client = chromadb.PersistentClient(path="./pokemon_db")

    # Start clean each run so re-running doesn't create duplicates.
    try:
        chroma_client.delete_collection(name="pokemon")
    except Exception:
        pass  # Collection didn't exist yet; fine.
    collection = chroma_client.create_collection(name="pokemon")

    ids, documents, metadatas = [], [], []

    print(f"Fetching {len(POKEMON_NAMES)} Pokemon...\n")
    for i, name in enumerate(POKEMON_NAMES, start=1):
        print(f"[{i}/{len(POKEMON_NAMES)}] {name}")
        result = build_pokemon_chunk(name)
        if result is None:
            continue
        chunk_text, metadata = result
        ids.append(name)
        documents.append(chunk_text)
        metadatas.append(metadata)
        time.sleep(0.3)  # Be polite to the free API; small pause between Pokemon.

    # Load everything into Chroma in one batch. Chroma embeds the documents
    # for us automatically using its default embedding model.
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"\nDone. {len(ids)} Pokemon stored in the 'pokemon' collection.")
    print("Here is one example chunk so you can see what got stored:\n")
    print(documents[0])

if __name__ == "__main__":
    main()