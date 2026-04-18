def load_and_chunk(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    return chunks

chunks = load_and_chunk("sample.txt")
for i, chunk in enumerate(chunks):
    print(f"[Chunk {i}]: {chunk}")