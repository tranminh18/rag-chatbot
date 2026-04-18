from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import faiss
import numpy as np

load_dotenv()
app = FastAPI()
client = Groq()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = open("sample.txt").read().split("\n\n")
chunks = [c.strip() for c in chunks if c.strip()]
vectors = embed_model.encode(chunks).astype("float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

class Question(BaseModel):
    text: str

@app.post("/chat")
def chat(q: Question):
    q_vec = embed_model.encode([q.text]).astype("float32")
    _, idx = index.search(q_vec, k=2)
    context = "\n".join([chunks[i] for i in idx[0]])

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQ: {q.text}"}]
    )
    return {"answer": res.choices[0].message.content}